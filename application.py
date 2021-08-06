from collections import OrderedDict
import streamlit as st
from streamlit.logger import get_logger
import views

LOGGER = get_logger(__name__)

# Dictionary of
# demo_name -> (demo_function, demo_description)
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
            "Portfolio Risk Analysis",
            (
                views.individual_cusip_analysis,
                """
This view will allow you to understand the risk of these CUSIPS.
""",
            ),
        ),
    ]
)


def main():
    view_name = st.sidebar.selectbox("Choose a view", list(VIEWS.keys()), 0)
    view = VIEWS[view_name][0]

    if view_name == "—":
        st.write("# Hello, KeyBank! 👋")
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
