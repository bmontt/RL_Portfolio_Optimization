import json
from sqlalchemy import create_engine
from models import Base

def create_tables():
    """
    Creates tables in the database based on the defined models.

    This function loads the database configuration from a JSON file,
    creates a database engine using the configuration, and then creates
    the tables in the database using the SQLAlchemy Base metadata.

    Note: The database configuration is expected to be stored in a file
    named 'config.json' in the same directory as this script.

    Returns:
        None
    """
    # Load database configuration
    with open('Rl_Portfolio_Optimization/config.json', 'r') as file:
        config = json.load(file)
    db_config = config['database']

    # Create database engine
    engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}")

    # Create tables
    Base.metadata.create_all(engine)

# Call the function to create tables
create_tables()
