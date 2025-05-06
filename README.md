MediAgent: AI Health Assistant 🧑‍⚕️🤖

MediAgent is an intelligent health assistant that provides personalized health advice through empathetic AI-driven conversations. Built with Streamlit, Python, and SQLite, MedAI offers a secure, scalable platform with features like JWT-based authentication, chat history, and input sanitization, ensuring a reliable experience for users.
Key Features:

    Authentication: Secure login and registration with JWT.

    Chat: Empathetic AI responses powered by a mock Gemini API (future medical tuning).

    Chat History: Storing and retrieving conversation history.

    Security: Input sanitization and rate limiting to ensure secure interactions.

    Responsive: Mobile-first design ensuring usability across all devices.

    Accessibility: Fully accessible with ARIA labels and high contrast for easy navigation.

Tech Stack:

    Frontend: Streamlit for interactive, web-based UI.

    Backend: Flask with JWT-based authentication, SQLite for data storage.

    Scalability: Future migration to PostgreSQL for scalability.

    Future Plans: Google/Facebook login, multi-language support, live doctor consultations.

Setup Instructions:
Backend Setup:

    Navigate to the backend directory:

cd backend

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Create a .env file and add:

GEMINI_API_KEY=your_api_key_here
JWT_SECRET=your-secret-key

Initialize SQLite:

sqlite3 database.db < ../database/schema.sql

Run Flask:

    python run.py

Frontend Setup (Streamlit):

    Install Streamlit:

pip install streamlit

Create a Streamlit script for the frontend (e.g., app.py):

streamlit run app.py

Start the development server:

    streamlit run app.py

Accessing the App:

    Streamlit Frontend: http://localhost:8501

    Backend API: http://localhost:5000

Scalability & Future Plans:

    Database: Move to PostgreSQL for larger data scale.

    Authentication: Integrate Google/Facebook login.

    Subscription Plans: Premium access for advanced features.

    Multi-language: Support for French, Arabic, Spanish (RTL).

    Mobile App: PWA or React Native app.

    AI Tuning: Future integration with RAG for enhanced medical accuracy.

    Medical Integration: Live doctor consultations via the platform.

License:

MIT License (see LICENSE file).
