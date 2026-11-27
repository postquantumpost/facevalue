---
applyTo: "**"
---

At the start of each new conversation, before answering the user, send a visible message: "Loaded (filepath)/LocalAGENTS.md".

Use the local python environment. Create it if needed with "python3 -m venv .venv". Start it
with "source .venv/bin/activate". Use "python3" instead of "/usr/bin/python3".

If it appears the remote application is showing an unexpecte screen, request guidance. I may be able
to reset the remote application state.

Always have OCR tests that match how OCR is used in the application including real tesseract invocations,
or other libraries that are used. Match the libraries that are used in the application in the tests.

