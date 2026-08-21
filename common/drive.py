"""Google Drive mounting — no-op outside Colab."""


def mount_drive(path="/content/drive"):
    """Mount Google Drive at `path`. No-op if not running in Colab."""
    try:
        from google.colab import drive
    except ImportError:
        return
    drive.mount(path)
