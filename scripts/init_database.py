import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils.models
from src.utils.database import init_db, engine
from src.utils.logger import logger
from sqlalchemy import text

def main():
    logger.info("Initializing database")

    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).fetchone()[0]
            logger.info(f"Connected: {version}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        sys.exit(1)

    try:
        init_db()

        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]

        if tables:
            logger.info(f"Tables created: {', '.join(tables)}")
        else:
            logger.error("No tables found - check model imports")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()