CREATE TABLE IF NOT EXISTS login_doctor (
    id_doctor SERIAL PRIMARY KEY,
    name_doctor VARCHAR(100) NOT NULL,
    date_of_birth_doctor DATE NOT NULL,
    email_doctor VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    gender VARCHAR(10),
    phone_number VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
