# Security

Report vulnerabilities through GitHub's private vulnerability reporting on this repository (Security → Report a vulnerability). Please do not open public issues for security reports.

Operational guidance: keep the API on loopback during development, supply credentials only through the environment variables in `.env.example`, and leave hosted synthesis disabled unless you have reviewed what data crosses that boundary. See [docs/security.md](docs/security.md) for the controls in place.
