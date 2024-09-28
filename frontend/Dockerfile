# Use the official Python 3.9 image from the Docker hub
# python:3.9
FROM python:3.10-slim 

# Create a directory named /app inside the container
WORKDIR /app

# Copy all the files from your current directory to /app in the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit uses, default is 8501
EXPOSE 8501

# Specify the default command to run Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Tell the container to run the streamlit.py file by default
CMD ["streamlit_app.py"]
