# Declares the web process for platforms such as Heroku.
# Gunicorn imports the Flask application object named "app" from app.py (app:app)
# and binds its HTTP listener to the platform-provided PORT environment variable.
web: gunicorn --bind $PORT app:app
