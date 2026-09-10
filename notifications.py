"""
SlopeSense — alert delivery.

When a station's risk crosses into High/Critical, `send_alert()` is called.
Right now it guarantees a console log line (so nothing is ever silently
dropped) and, if ALERT_WEBHOOK_URL is set as an environment variable, also
POSTs a JSON payload to that URL — this lets you wire it into something
real (a Slack/Discord incoming webhook, a Zapier/n8n flow, your own
notification microservice) without changing any code.

For a production deployment, replace the webhook call with a direct
integration: Twilio or AWS SNS for SMS to on-call staff/local authorities,
SES/SendGrid for email, or a push notification service for a mobile app.
"""

import os
import json
import urllib.request

WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()


def send_alert(station_name: str, level: str, message: str) -> dict:
    """Deliver an alert. Always logs locally; optionally forwards to a webhook."""
    text = f"[SlopeSense ALERT] {station_name}: {message}"
    print(text)

    if not WEBHOOK_URL:
        return {"delivered_via": "console_log", "webhook_configured": False}

    payload = json.dumps({"text": text, "station": station_name, "level": level}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=3)
        return {"delivered_via": "webhook", "webhook_configured": True}
    except Exception as e:
        print(f"  (webhook delivery failed: {e})")
        return {"delivered_via": "console_log", "webhook_configured": True, "error": str(e)}
