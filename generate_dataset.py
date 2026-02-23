import pandas as pd
import random

fake_sentences = [
    "Aliens landed in India yesterday",
    "Government secretly giving money to everyone",
    "Drinking only coffee cures all diseases",
    "Earth will end tomorrow according to scientists",
    "Phones are spying on every human 24/7",
    "Scientists confirm dinosaurs are still alive",
    "Secret island discovered with magical powers",
    "Eating chocolate increases IQ instantly"
]

real_sentences = [
    "India launched a new satellite successfully",
    "Government announced new education policy",
    "Doctors recommend regular exercise for health",
    "New technology released for renewable energy",
    "Scientists discovered new species in ocean",
    "Stock market shows steady growth this year",
    "New highway project approved by parliament",
    "Medical researchers developed new vaccine"
]

fake_data = []
real_data = []

for i in range(1000):
    fake_data.append(["Fake News " + str(i), random.choice(fake_sentences), "News", "2024-01-01"])

for i in range(1000):
    real_data.append(["Real News " + str(i), random.choice(real_sentences), "News", "2024-01-01"])

fake_df = pd.DataFrame(fake_data, columns=["title", "text", "subject", "date"])
real_df = pd.DataFrame(real_data, columns=["title", "text", "subject", "date"])

fake_df.to_csv("fake.csv", index=False)
real_df.to_csv("true.csv", index=False)

print("Large dataset generated successfully!")