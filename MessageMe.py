from twilio.rest import Client
import GlobalVars as gv

def send_message(message):
    client = Client(gv.twilio_ID, gv.twilio_secret_ID)
    client.messages.create(to=gv.twilio_to,
                           from_=gv.twilio_from,
                           body=message)