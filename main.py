import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import classification_report, accuracy_score

data = {
    
    "text": [
        "it was good",
        "I love flying with @SouthwestAir! Great service 😊",
        "Terrible experience with @united — delayed for 5 hours and no help!",
        "Thanks @Delta for the upgrade to first class ✈️",
        "Why is @AmericanAir always late? 😤 #frustrated",
        "No complaints today. Smooth flight with JetBlue. 👍",
        "Cancelled my flight without notice. Disappointed. @united",
        "Amazing crew and on-time arrival @AlaskaAir 👏👏",
        "Where is my baggage? @AmericanAir",
        "Customer support was helpful today. Thank you @Delta",
        "Not the best meal, but decent flight overall. @SouthwestAir",
        "Horrible service, never flying again",
        "Flight attendants were rude",
        "Excellent in-flight entertainment and comfy seats",
        "We landed early! Great experience.",
        "Luggage was lost and no one helped",
        "Very polite staff and fast boarding",
        "Awful check-in process",
        "Smooth takeoff and very friendly pilot",
        "The snacks were stale and cold"
    ],
    "airline_sentiment": [
        "positive", "positive", "negative", "positive", "negative",
        "neutral", "negative", "positive", "negative", "positive",
        "neutral", "negative", "negative", "positive", "positive",
        "negative", "positive", "negative", "positive", "negative"
    ]
}


df = pd.DataFrame(data)
df.dropna(inplace=True)


st.title("✈️ Airline Sentiment Analyzer")
st.markdown("This app predicts the sentiment of airline feedback.")


with st.expander("Show Dataset"):
    st.dataframe(df)


st.subheader("Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(x='airline_sentiment', data=df, ax=ax)
st.pyplot(fig)


st.subheader("Sentiment Word Clouds")
for sentiment in ['positive', 'negative', 'neutral']:
    text = " ".join(df[df['airline_sentiment'] == sentiment]['text'])
    wc = WordCloud(background_color='white', max_words=200).generate(text)
    st.markdown(f"**{sentiment.capitalize()} Feedback**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


X = df['text']
y = df['airline_sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vect, y_train)


st.subheader("Model Evaluation")
y_pred = model.predict(X_test_vect)
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))


st.subheader("🔍 Predict Sentiment from Feedback")
user_input = st.text_area("✍️ Enter your airline feedback here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some feedback text.")
    else:
        user_vect = vectorizer.transform([user_input])
        prediction = model.predict(user_vect)[0]
        st.success(f"🧠 Predicted Sentiment: **{prediction.capitalize()}**")

