# 1. Imagen base con Python 3.11 (ligera)
FROM python:3.11-slim

# 2. Establece directorio de trabajo
WORKDIR /app

# 3. Copia requirements y los instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia el resto de la aplicación
COPY . .

# 5. Expone el puerto 8000
EXPOSE 8000

# 6. Comando por defecto para arrancar la app con Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
