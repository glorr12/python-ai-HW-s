
import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TICKETS = {
    "booking": [
        "I want to cancel my reservation for next weekend",
        "Can I change the check-in date of my booking?",
        "The host rejected my booking request without explanation",
        "How do I extend my stay by two more nights?",
        "My booking is still pending after 24 hours",
        "I booked the wrong apartment by mistake, please help",
        "Is it possible to add one more guest to my reservation?",
        "The booking confirmation email never arrived",
        "Can I book a flat for three months?",
        "Why was my reservation cancelled automatically?",
        "I need to shorten my stay and leave two days earlier",
        "The calendar shows the dates are free but I cannot reserve them",
        "How far in advance can I make a reservation?",
        "Can I transfer my booking to a friend?",
        "I arrived and the host says there is no reservation under my name",
        "The check-in time in my booking is wrong",
        "How do I see all my upcoming reservations?",
        "I got two confirmations for the same stay",
        "Can I reserve two apartments at the same time?",
        "My host asked me to cancel the booking myself",
        "What happens if I miss my check-in?",
        "I want to move my reservation to another month",
        "The app crashed while I was confirming the booking",
        "Can I book without the host approving it first?",
        "My trip dates changed, how do I update the reservation?",
        "The number of nights in my booking is incorrect",
        "Is there a minimum stay for this listing?",
        "I cannot find my reservation in the app anymore",
        "The host wants to change my check-out date",
        "How do I request an early check-in?",
    ],
    "payment": [
        "I was charged twice for the same stay",
        "When will I get my refund after cancelling?",
        "My credit card was declined at checkout",
        "Can I pay with PayPal?",
        "The total price is higher than shown in the search",
        "I need an invoice for my company",
        "Why is there a cleaning fee added to my bill?",
        "The security deposit was not returned",
        "Can I split the payment between two cards?",
        "I paid but the booking still says unpaid",
        "What currency will I be charged in?",
        "The service fee seems too high",
        "My refund is less than I expected",
        "How do I add a new payment method?",
        "I see an unknown charge on my bank statement",
        "Can I pay in cash when I arrive?",
        "The discount code did not apply to the price",
        "When is the money taken from my account?",
        "I was charged after cancelling for free",
        "Is VAT included in the price?",
        "The payment page keeps showing an error",
        "Can I get a receipt for last month's rent?",
        "How long does a bank transfer take to be confirmed?",
        "Host asks me to pay outside the platform, is it safe?",
        "I want to pay the rent monthly instead of upfront",
        "My payment is stuck in processing",
        "The exchange rate you used is wrong",
        "How much is the deposit for this apartment?",
        "I need to update the billing address on my invoice",
        "Why did my card get a temporary hold?",
    ],
    "listing": [
        "How do I publish a new apartment on the site?",
        "My listing is not showing up in search results",
        "I cannot upload photos to my listing",
        "How do I change the price per night for my flat?",
        "Can I temporarily hide my listing?",
        "The address on my listing is displayed incorrectly",
        "How do I add amenities like wifi and parking?",
        "My listing was removed without notice",
        "Can I list a room instead of the whole apartment?",
        "How do I set different prices for weekends?",
        "The map pin for my property is in the wrong place",
        "How many photos can I add to one listing?",
        "I want to delete my old listing completely",
        "Can I copy an existing listing for a second flat?",
        "The description of my house was cut off",
        "How do I block dates when the flat is not available?",
        "Can I set a minimum number of nights for my property?",
        "My listing has the wrong number of bedrooms",
        "How do I add house rules to the listing?",
        "Search filters do not find my apartment by district",
        "I want to change the title of my listing",
        "Can I publish a listing in two languages?",
        "How do I mark my apartment as pet friendly?",
        "The listing status is stuck on under review",
        "How do I set the check-in instructions for guests?",
        "Photos in my listing appear rotated",
        "Can I offer a weekly discount on my property?",
        "My listing shows another host's phone number",
        "How do I sort my listings by date created?",
        "I cannot edit the property type of my listing",
    ],
    "account": [
        "I forgot my password and cannot log in",
        "How do I change the email on my account?",
        "My account was blocked, why?",
        "I did not receive the verification code",
        "How do I delete my account and all my data?",
        "Can I switch my profile from guest to host?",
        "Two-factor authentication code is not working",
        "Someone else logged into my account",
        "How do I change my profile photo?",
        "The login page says my email is not registered",
        "I want to change my phone number",
        "How do I verify my identity?",
        "My account shows the wrong name",
        "I registered twice with different emails, can you merge them?",
        "The password reset link has expired",
        "How do I turn off email notifications?",
        "I cannot log in with Google anymore",
        "Can I have one account for guest and host roles?",
        "How do I download all my personal data?",
        "The app logs me out every few minutes",
        "My email confirmation link does not work",
        "How can I change the language of my profile?",
        "I suspect my account was hacked",
        "Why do I need to upload my ID document?",
        "How do I update my date of birth?",
        "The sign up form gives an error",
        "I lost access to the phone linked to my account",
        "How do I log out from all devices?",
        "Can I change my username?",
        "My profile says it is not verified",
    ],
    "review": [
        "The host left an unfair review about me",
        "How do I write a review after my stay?",
        "Can I edit the review I left yesterday?",
        "My review is not visible on the listing page",
        "A guest wrote a fake review on my apartment",
        "How long do I have to leave a review?",
        "Can I delete a negative review?",
        "The rating of my listing dropped for no reason",
        "Can I reply to a review from a guest?",
        "I want to report an offensive review",
        "Why can't I rate the host?",
        "My average rating is calculated incorrectly",
        "Can hosts see my review before publishing theirs?",
        "The review mentions personal information about me",
        "I stayed there but the site says I cannot leave feedback",
        "How are the star ratings calculated?",
        "My five star review disappeared",
        "A competitor is posting bad reviews on my listing",
        "Can I leave a review for a cancelled stay?",
        "I want to add photos to my review",
        "The host threatened me because of my review",
        "Do reviews affect my position in search?",
        "My review was published under another name",
        "Can I hide reviews older than two years?",
        "The guest review contains false information",
        "Why was my review rejected by moderation?",
        "Can I review a listing anonymously?",
        "The feedback form does not open",
        "How do I see reviews I have written?",
        "The review says the flat was dirty but it was not",
    ],
}

TYPOS = {"reservation": "resrvation", "payment": "paymnet", "password": "pasword",
         "listing": "lisitng", "review": "reveiw", "cancel": "cancell"}


def add_noise(rows, rng):
    noisy = []
    for text, label in rows:
        r = rng.random()
        if r < 0.10:
            text = "  " + text.upper() + "   "          # регистр + пробелы
        elif r < 0.18:
            text = f"<p>{text}</p>"                   # HTML-мусор
        elif r < 0.30:
            for good, bad in TYPOS.items():           # опечатки
                text = text.replace(good, bad)
        if rng.random() < 0.15:
            label = rng.choice([label.upper(), f" {label} ", label.capitalize()])
        noisy.append((text, label))

    duplicates = rng.sample(noisy, 20)                # 20 дубликатов
    noisy += [(t.lower() if i % 2 else t, l) for i, (t, l) in enumerate(duplicates)]
    noisy += [("", "booking"), ("   ", "payment"), ("Hello?", ""), ("test test", "")]  # мусор
    rng.shuffle(noisy)
    return noisy


def main() -> Path:
    rng = random.Random(7)
    rows = [(text, label) for label, texts in TICKETS.items() for text in texts]
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "raw_tickets.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(add_noise(rows, rng))
    print(f"Сохранено: {path}")
    return path


if __name__ == "__main__":
    main()
