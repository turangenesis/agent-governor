FROM python:3.12-slim

WORKDIR /app

# Copy the whole project first so the editable install can see pyproject.toml + src/.
# (requirements.txt installs the package itself via `-e .[demo]`, not just its deps,
# so `import governor` resolves at startup after the src/ layout move.)
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Streamlit must bind 0.0.0.0 for Akash to route to it.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
