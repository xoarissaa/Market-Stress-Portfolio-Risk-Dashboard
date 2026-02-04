import sqlite3
import json
from datetime import datetime

class PortfolioDB:
    """
    Simple SQLite Wrapper for Portfolio Persistence.
    Stores portfolios as JSON blobs.
    """
    
    def __init__(self, db_path="portfolios.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT,
                assets_json TEXT,
                notes TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_portfolio(self, name, assets_dict, notes=""):
        """
        Saves a portfolio to the DB.
        :param name: Name of the portfolio (e.g. "My 60/40")
        :param assets_dict: Dict of {Ticker: Weight} (e.g. {'SPY': 0.6, 'TLT': 0.4})
        :param notes: Optional text notes
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Check if name exists, if so update?
        # For simplicity, we'll just insert new entries, user can delete old ones.
        # Or better: Unique constraint on name? Let's keep it simple (allow duplicates)
        
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        assets_json = json.dumps(assets_dict)
        
        c.execute('INSERT INTO portfolios (name, created_at, assets_json, notes) VALUES (?, ?, ?, ?)',
                  (name, created_at, assets_json, notes))
        
        conn.commit()
        conn.close()

    def load_portfolios(self):
        """
        Returns a list of all portfolios.
        :return: List of dicts [{'id': 1, 'name': '...', 'assets': {...}, ...}]
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        c = conn.cursor()
        
        c.execute('SELECT * FROM portfolios ORDER BY created_at DESC')
        rows = c.fetchall()
        
        portfolios = []
        for row in rows:
            portfolios.append({
                'id': row['id'],
                'name': row['name'],
                'created_at': row['created_at'],
                'assets': json.loads(row['assets_json']),
                'notes': row['notes']
            })
            
        conn.close()
        return portfolios

    def delete_portfolio(self, portfolio_id):
        """Deletes a portfolio by ID."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM portfolios WHERE id = ?', (portfolio_id,))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    # Test
    db = PortfolioDB()
    db.save_portfolio("Test Portfolio", {"SPY": 0.6, "TLT": 0.4}, "Test Note")
    print(db.load_portfolios())
