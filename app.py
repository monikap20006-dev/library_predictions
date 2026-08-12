import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Library Management System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    border: 1px solid #ddd;
    text-align: center;
}

.result {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "Books_data - Sheet1.csv"
    )

    return data


@st.cache_resource
def load_model():

    rf_model = joblib.load(
        "library_random_forest.pkl"
    )

    encoders = joblib.load(
        "library_encoders.pkl"
    )

    return rf_model, encoders


# ============================================================
# LOAD
# ============================================================

try:

    df = load_data()

    model, encoders = load_model()

except Exception as e:

    st.error("Unable to load the required files.")

    st.code(str(e))

    st.stop()


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "bid",
    "title",
    "author",
    "category",
    "status"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "Missing columns in CSV: "
        + str(missing_columns)
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 Library Management System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Random Forest Based Book Availability Prediction</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 Library Menu")

menu = st.sidebar.radio(
    "Select Option",
    [
        "Book Prediction",
        "Library Dataset",
        "Statistics"
    ]
)


# ============================================================
# BOOK PREDICTION PAGE
# ============================================================

if menu == "Book Prediction":

    st.header("🔍 Book Availability Prediction")

    st.write(
        "Select a book from the library dataset "
        "to check its availability."
    )

    st.divider()

    # --------------------------------------------------------
    # BOOK SELECTION
    # --------------------------------------------------------

    book_list = df[
        "title"
    ].dropna().unique().tolist()

    selected_title = st.selectbox(
        "📖 Select Book",
        book_list
    )

    # --------------------------------------------------------
    # GET BOOK
    # --------------------------------------------------------

    selected_rows = df[
        df["title"] == selected_title
    ]

    selected_book = selected_rows.iloc[0]

    bid = selected_book["bid"]
    author = selected_book["author"]
    category = selected_book["category"]
    actual_status = selected_book["status"]

    # --------------------------------------------------------
    # BOOK INFORMATION
    # --------------------------------------------------------

    st.subheader("📋 Book Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Book ID",
            str(bid)
        )

    with col2:

        st.metric(
            "Author",
            str(author)
        )

    with col3:

        st.metric(
            "Category",
            str(category)
        )

    with col4:

        st.metric(
            "Dataset Status",
            str(actual_status).upper()
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Book Status",
        type="primary",
        use_container_width=True
    ):

        try:

            # Encode values using saved encoders

            title_encoded = encoders[
                "title"
            ].transform(
                [str(selected_title)]
            )[0]

            author_encoded = encoders[
                "author"
            ].transform(
                [str(author)]
            )[0]

            category_encoded = encoders[
                "category"
            ].transform(
                [str(category)]
            )[0]

            # ------------------------------------------------
            # CREATE MODEL INPUT
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "bid": [
                    bid
                ],

                "title": [
                    title_encoded
                ],

                "author": [
                    author_encoded
                ],

                "category": [
                    category_encoded
                ]

            })

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                input_data
            )[0]

            # ------------------------------------------------
            # CONVERT PREDICTION
            # ------------------------------------------------

            status_encoder = encoders.get(
                "status"
            )

            if status_encoder is not None:

                predicted_status = (
                    status_encoder
                    .inverse_transform(
                        [prediction]
                    )[0]
                )

            else:

                if prediction == 0:
                    predicted_status = "available"
                else:
                    predicted_status = "issued"

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🎯 Prediction Result"
            )

            if str(
                predicted_status
            ).lower() == "available":

                st.success(
                    "✅ BOOK IS AVAILABLE"
                )

            else:

                st.error(
                    "❌ BOOK IS ISSUED"
                )

            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.write(
                    "**ML Prediction:**"
                )

                st.info(
                    str(
                        predicted_status
                    ).upper()
                )

            with result_col2:

                st.write(
                    "**Actual Dataset Status:**"
                )

                if str(
                    actual_status
                ).lower() == "available":

                    st.success(
                        str(
                            actual_status
                        ).upper()
                    )

                else:

                    st.error(
                        str(
                            actual_status
                        ).upper()
                    )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# DATASET PAGE
# ============================================================

elif menu == "Library Dataset":

    st.header("📊 Library Books")

    st.write(
        "Complete library dataset"
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STATISTICS PAGE
# ============================================================

elif menu == "Statistics":

    st.header("📈 Library Statistics")

    total_books = len(df)

    available_books = len(
        df[
            df["status"]
            .astype(str)
            .str.lower()
            == "available"
        ]
    )

    issued_books = len(
        df[
            df["status"]
            .astype(str)
            .str.lower()
            == "issued"
        ]
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 Total Books",
            total_books
        )

    with col2:

        st.metric(
            "✅ Available",
            available_books
        )

    with col3:

        st.metric(
            "❌ Issued",
            issued_books
        )

    st.divider()

    # --------------------------------------------------------
    # CATEGORY COUNT
    # --------------------------------------------------------

    st.subheader(
        "📂 Books by Category"
    )

    category_count = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category_count.columns = [
        "Category",
        "Number of Books"
    ]

    st.dataframe(
        category_count,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # STATUS COUNT
    # --------------------------------------------------------

    st.subheader(
        "📌 Book Status"
    )

    status_count = (
        df["status"]
        .value_counts()
        .reset_index()
    )

    status_count.columns = [
        "Status",
        "Number of Books"
    ]

    st.dataframe(
        status_count,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Library Management System | "
    "Random Forest Machine Learning"
)
