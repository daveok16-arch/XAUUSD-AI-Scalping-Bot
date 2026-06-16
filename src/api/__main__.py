"""
Entry point for running the API directly with: python -m src.api
"""

import os
import sys

from src.api.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
