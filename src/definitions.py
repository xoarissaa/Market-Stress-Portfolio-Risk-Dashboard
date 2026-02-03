class RegimeDefinitions:
    """
    Centralized vocabulary for Market Regimes.
    Designed to be professional, descriptive, and non-alarmist.
    """
    
    LABELS = {
        0: "Stable Growth",      # Green
        1: "Elevated Uncertainty", # Yellow
        2: "High Stress"         # Red
    }
    
    COLORS = {
        0: "normal",   # Green-ish in Streamlit
        1: "off",      # Grey/Yellow-ish
        2: "inverse"   # Red-ish
    }

    DESCRIPTIONS = {
        0: "Markets are behaving within normal historical ranges. Volatility is low, and the general trend is positive.",
        1: "Markets are digesting new information. Prices are swinging more widely as investors reassess risk/reward.",
        2: "Markets are reacting to significant stress. Expect sharp daily moves and higher correlation between assets."
    }

    INVESTOR_CONTEXT = {
        0: "**Long-Term View:** This is a constructive environment where compounding typically works best. The 'road is clear'.",
        1: "**Long-Term View:** Noise levels are high. It is normal for trends to stall or reverse temporarily here.",
        2: "**Long-Term View:** This is a defensive environment. Your portfolio's structural hedges (e.g., Gold, Treasuries) are most important now."
    }

    @staticmethod
    def get_regime_info(regime_code):
        """Returns a tuple of (Label, Description, Context) for a given code."""
        return (
            RegimeDefinitions.LABELS.get(regime_code, "Unknown"),
            RegimeDefinitions.DESCRIPTIONS.get(regime_code, "No description available."),
            RegimeDefinitions.INVESTOR_CONTEXT.get(regime_code, "No context available.")
        )
