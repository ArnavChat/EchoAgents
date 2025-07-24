# EchoAgents
The EchoAgents project aims to prove that true AI autonomy comes from a robust, stateful, and observable system architecture that orchestrates LLMs, rather than from the power of the language model alone.
Here's a clean and beginner-friendly `README.md` you can use to help others set up and run your Docker-based Python app:

---

````markdown

## 🚀 Prerequisites

Before you begin, make sure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

To verify installations:
```bash
docker --version
docker-compose --version
````

---

## 🛠️ Setup & Run

### 1. Clone the Repository

```bash
git clone https://github.com/ArnavChat/EchoAgents.git
cd echoagents
```

### 2. Build the Docker Image

```bash
docker-compose build
```

> 💡 This installs all necessary Python packages listed in `requirements.txt`.

### 3. Run the App

```bash
docker-compose up
```

The app will now start inside a container and execute `test.py`. You should see output directly in your terminal.

---

## 📂 Project Structure

```
echoagents/
│
├── Dockerfile              # Builds the Docker image
├── docker-compose.yml      # Defines services and container behavior
├── requirements.txt        # Python dependencies
├── test.py                 # Entry-point script
└── (other Python modules and files)
```

---

## 🐳 Docker Commands (Quick Reference)

* **Rebuild the image from scratch:**

  ```bash
  docker-compose build --no-cache
  ```

* **Stop and remove running containers:**

  ```bash
  docker-compose down
  ```

* **Access the container shell (for debugging):**

  ```bash
  docker-compose run backend bash
  ```

---

