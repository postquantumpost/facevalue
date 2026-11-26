# facevalue
Face metics multitool

## Remote desktop

Use `--desktop <login.json>` to connect with AsyncVNC and display the remote
desktop at up to 640 by 480 pixels. The full desktop remains visible with its
aspect ratio preserved. The initial capture is followed by another capture
every 10 seconds. At startup, the application view is verified by OCR: the
expected `Cosmetics` heading is checked first, followed by `Alter Face & Hair`.
Mouse control is then checked over the same VNC connection by selecting
`Adjust Face Template` and verifying the resulting
`Alter Face & Hair > Adjust Face Template` heading. If any verification fails,
the program displays a message and exits. The `tesseract` executable must be
installed and available on `PATH`.

The login file requires `host`. `port` defaults to 5900, while `username` and
`password` are optional because the required authentication fields depend on
the VNC server:

```json
{
  "host": "desktop.example.test",
  "port": 5900,
  "username": "operator",
  "password": "replace-with-vnc-password"
}
```

For ordinary VNC password authentication, `host`, `port`, and `password` are
normally sufficient. Include `username` for Apple Remote Desktop or another
server mode that requires it. Treat this file as a credential file and do not
commit it to source control.
