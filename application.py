from collections import OrderedDict
import streamlit as st
from streamlit.logger import get_logger
import views

LOGGER = get_logger(__name__)

# D:\Python\Python38\Scripts\streamlit run D:\Python\Projects\bondProjectLocal\application.py
# D:\Python\bondProjectLocal\Scripts\streamlit run D:\Python\Projects\bondProjectLocal\application.py

# Dictionary of
VIEWS = OrderedDict(
    [
        ("—", (views.intro, None)),
        (
            "P&L CUSIP Attribution",
            (
                views.pnl_cusip_attribution,
                """
This view will allow you to find which CUSIPS are most significant to the portfolio's overall profit and loss.
""",
            ),
        ),
        (
            "Individual CUSIP Analysis",
            (
                views.individual_cusip_analysis,
                """
This view will allow you to analyze characteristics of specific CUSIPS.
""",
            ),
        ),
        (
            "Bucket and Characteristic Analysis",
            (
                views.bucket_industry,
                """
This view will allow you to compare performance across various characteristics for the securities.
""",
            ),
        ),
        (
            "Unrelated: Merchant Name Extraction",
            (
                views.merchant_extraction,
                """
This view is unrelated to bond analysis. This was something I worked on (mainly out of boredom and curiosity) after
meeting with Cole. I am not sure if what mechanism is in place to extract merchant names from raw statements--but I hope
that this is an insightful example of what is possible!

This view will allow you to interact with a convolutional
neural network I trained on my own credit statements to extract merchant names from raw transaction strings. 
This is a scalable solution and will only get better with more data!
""",
            ),
        ),
    ]
)


def main():
    view_name = st.sidebar.selectbox("Choose a view", list(VIEWS.keys()), 0)
    view = VIEWS[view_name][0]

    if view_name == "—":
        st.write("# Hello, KeyBanc! 👋")
    else:
        st.markdown("# %s" % view_name)
        description = VIEWS[view_name][1]
        if description:
            st.write(description)
        # Clear everything from the intro page.
        # We only have 4 elements in the page so this is intentional overkill.
        for i in range(10):
            st.empty()

    view()


if __name__ == "__main__":
    main()
