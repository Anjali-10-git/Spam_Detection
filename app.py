import re
import string
import pickle
import joblib
import streamlit as st


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "Spam_Detection.pkl"

# If your training dataset uses:
# 1 = Spam, 0 = Ham
# keep this as 1.
#
# If your dataset uses:
# 0 = Spam, 1 = Ham
# change this to 0.
SPAM_LABEL = 1


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Spam Detection App",
    page_icon="📧",
    layout="centered"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def wordopt(text):

    text = str(text).lower()

    # Keep URLs as a token.
    # Removing every URL can remove useful spam information.
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' URL ',
        text
    )

    # Replace email addresses
    text = re.sub(
        r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',
        ' EMAIL ',
        text
    )

    # Remove HTML
    text = re.sub(
        r'<[^>]*>',
        ' ',
        text
    )

    # Keep words and numbers
    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    # Remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ============================================================
# SPAM SIGNAL DETECTOR
# ============================================================

def spam_score(text):

    text_lower = text.lower()

    score = 0
    reasons = []

    # --------------------------------------------------------
    # Money / prizes
    # --------------------------------------------------------

    money_words = [
        "cash",
        "money",
        "prize",
        "reward",
        "winner",
        "won",
        "winning",
        "jackpot",
        "lottery",
        "free",
        "$",
        "usd"
    ]

    found_money = [
        word for word in money_words
        if word in text_lower
    ]

    if found_money:
        score += min(len(found_money) * 2, 6)
        reasons.append("Prize / money language")


    # --------------------------------------------------------
    # Urgency
    # --------------------------------------------------------

    urgency_words = [
        "urgent",
        "act now",
        "immediately",
        "hurry",
        "limited time",
        "expires",
        "today",
        "within 24 hours",
        "last chance",
        "claim now",
        "click now"
    ]

    found_urgency = [
        word for word in urgency_words
        if word in text_lower
    ]

    if found_urgency:
        score += min(len(found_urgency) * 2, 6)
        reasons.append("Urgent / pressure language")


    # --------------------------------------------------------
    # Suspicious actions
    # --------------------------------------------------------

    action_words = [
        "click here",
        "click now",
        "claim your",
        "claim prize",
        "verify your account",
        "confirm your account",
        "login now",
        "get your reward",
        "collect your reward"
    ]

    found_actions = [
        word for word in action_words
        if word in text_lower
    ]

    if found_actions:
        score += min(len(found_actions) * 2, 6)
        reasons.append("Suspicious call-to-action")


    # --------------------------------------------------------
    # Suspicious links
    # --------------------------------------------------------

    if re.search(r'https?://|www\.', text_lower):

        score += 3
        reasons.append("Contains a URL")


    # --------------------------------------------------------
    # Excessive exclamation marks
    # --------------------------------------------------------

    exclamation_count = text.count("!")

    if exclamation_count >= 3:

        score += 2
        reasons.append("Excessive exclamation marks")


    # --------------------------------------------------------
    # ALL CAPS words
    # --------------------------------------------------------

    uppercase_words = re.findall(
        r'\b[A-Z]{4,}\b',
        text
    )

    if len(uppercase_words) >= 2:

        score += 2
        reasons.append("Excessive uppercase wording")


    return score, reasons


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    try:
        return joblib.load(MODEL_PATH)

    except Exception:

        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        f"""
        ❌ Could not load `{MODEL_PATH}`

        Error:

        {e}
        """
    )


# ============================================================
# TITLE
# ============================================================

st.title("📧 Spam Detection App")

st.write(
    "Enter an email or SMS message and check whether it is "
    "**Spam** or **Not Spam (Ham)**."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

if model_loaded:

    with st.expander("🔧 Model Information"):

        st.write("Model type:")

        st.code(str(type(model)))

        if hasattr(model, "classes_"):

            st.write("Model classes:")

            st.code(str(model.classes_))

        elif hasattr(model, "named_steps"):

            st.write("Pipeline steps:")

            st.code(
                str(list(model.named_steps.keys()))
            )

            try:

                classifier = model.steps[-1][1]

                if hasattr(classifier, "classes_"):

                    st.write("Classifier classes:")

                    st.code(
                        str(classifier.classes_)
                    )

            except Exception:
                pass


# ============================================================
# TEXT BOX
# ============================================================

user_input = st.text_area(
    "Enter message text:",
    height=200,
    placeholder=(
        "Example: Congratulations! You won $5000. "
        "Click here to claim your prize!"
    )
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    predict_clicked = st.button(
        "🔍 Predict",
        use_container_width=True,
        disabled=not model_loaded
    )

with col2:

    clear_clicked = st.button(
        "🧹 Clear",
        use_container_width=True
    )


if clear_clicked:

    st.rerun()


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    if not user_input.strip():

        st.warning(
            "⚠️ Please enter a message first."
        )

    else:

        # ----------------------------------------------------
        # Clean message
        # ----------------------------------------------------

        cleaned_text = wordopt(user_input)


        # ----------------------------------------------------
        # Spam rules
        # ----------------------------------------------------

        rule_score, reasons = spam_score(user_input)


        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        try:

            prediction = model.predict(
                [cleaned_text]
            )[0]


            # -----------------------------------------------
            # Determine ML prediction
            # -----------------------------------------------

            if isinstance(prediction, str):

                prediction_lower = prediction.lower().strip()

                if prediction_lower in [
                    "spam",
                    "junk"
                ]:

                    ml_is_spam = True

                elif prediction_lower in [
                    "ham",
                    "not spam",
                    "not_spam",
                    "legitimate"
                ]:

                    ml_is_spam = False

                else:

                    # Unknown string label
                    ml_is_spam = False

            else:

                # Numeric classification
                ml_is_spam = (
                    prediction == SPAM_LABEL
                )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    [cleaned_text]
                )[0]

                classes = list(
                    model.classes_
                )

                try:

                    index = classes.index(
                        prediction
                    )

                    confidence = probabilities[index]

                except Exception:

                    confidence = max(
                        probabilities
                    )


            # ------------------------------------------------
            # FINAL DECISION
            # ------------------------------------------------
            #
            # If the ML model says spam -> SPAM
            #
            # OR
            #
            # If the message contains many strong spam
            # characteristics -> SPAM
            #
            # This prevents obvious spam from being shown
            # as Ham when the existing model is weak.
            # ------------------------------------------------

            if ml_is_spam:

                final_is_spam = True

                decision_reason = (
                    "The machine-learning model classified "
                    "this message as spam."
                )

            elif rule_score >= 6:

                final_is_spam = True

                decision_reason = (
                    "The message contains several strong "
                    "spam characteristics."
                )

            else:

                final_is_spam = False

                decision_reason = (
                    "The machine-learning model classified "
                    "the message as Ham and the spam-signal "
                    "score was low."
                )


            # =================================================
            # RESULT
            # =================================================

            if final_is_spam:

                st.error(
                    "🚨 This message looks like **SPAM**."
                )

            else:

                st.success(
                    "✅ This message looks like "
                    "**Not Spam (Ham)**."
                )


            # -------------------------------------------------
            # Details
            # -------------------------------------------------

            st.write(
                f"**Spam signal score:** {rule_score}"
            )

            st.info(
                decision_reason
            )


            # -------------------------------------------------
            # Confidence
            # -------------------------------------------------

            if confidence is not None:

                st.write(
                    f"ML confidence: "
                    f"**{confidence * 100:.2f}%**"
                )


            # =================================================
            # DEBUG INFORMATION
            # =================================================

            with st.expander(
                "🔍 See prediction details"
            ):

                st.write(
                    "**Raw model prediction:**"
                )

                st.code(
                    str(prediction)
                )

                st.write(
                    "**ML says spam:**"
                )

                st.code(
                    str(ml_is_spam)
                )

                st.write(
                    "**Spam rule score:**"
                )

                st.code(
                    str(rule_score)
                )

                if reasons:

                    st.write(
                        "**Detected spam signals:**"
                    )

                    for reason in reasons:

                        st.write(
                            f"• {reason}"
                        )


            # =================================================
            # CLEANED TEXT
            # =================================================

            with st.expander(
                "See cleaned text sent to model"
            ):

                st.code(
                    cleaned_text
                    if cleaned_text
                    else "(empty)"
                )


        except Exception as e:

            st.error(
                f"❌ Prediction failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Spam Detection System · "
    "Machine Learning + Spam Signal Analysis"
)