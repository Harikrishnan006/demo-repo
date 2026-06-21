import base64
import pandas as pd
import streamlit as st


def clean_dataset(df: pd.DataFrame, drop_all_missing: bool, fill_strategy: str, drop_duplicates: bool) -> pd.DataFrame:
    cleaned = df.copy()

    if drop_all_missing:
        cleaned = cleaned.dropna(how="all")

    if fill_strategy == "Median / Mode":
        for column in cleaned.columns:
            if cleaned[column].dtype.kind in "biufc":
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                cleaned[column] = cleaned[column].fillna(cleaned[column].mode().iloc[0] if not cleaned[column].mode().empty else "")
    elif fill_strategy == "Forward fill":
        cleaned = cleaned.fillna(method="ffill").fillna(method="bfill")

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    return cleaned


def get_download_data(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="CSV Analyzer", page_icon="📊", layout="wide")
    st.title("CSV Analyzer and Cleaner")
    st.write(
        "Upload a CSV file to inspect the data, run a few basic analyses, visualize key columns, "
        "and download a cleaned version of the dataset."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV to begin analysis.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        return

    st.sidebar.header("Cleaning options")
    drop_all_missing = st.sidebar.checkbox("Drop rows with all missing values", value=True)
    fill_strategy = st.sidebar.radio(
        "Fill missing values",
        options=["None", "Median / Mode", "Forward fill"],
        index=1,
    )
    drop_duplicates = st.sidebar.checkbox("Drop duplicate rows", value=True)

    cleaned_df = clean_dataset(df, drop_all_missing, fill_strategy, drop_duplicates)

    st.header("Data preview")
    st.metric("Rows", f"{df.shape[0]:,}")
    st.metric("Columns", f"{df.shape[1]:,}")
    st.dataframe(df.head(10), use_container_width=True)

    st.header("Structure & summary")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]), use_container_width=True)

    with col2:
        st.subheader("Missing values")
        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "missing_count"]
        st.dataframe(missing, use_container_width=True)

    st.subheader("Basic statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    if not df.empty:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_columns:
            st.header("Charts")
            summary = df[numeric_columns].agg(["mean", "median", "std"]).T
            summary.columns = ["Mean", "Median", "Std"]
            st.subheader("Numeric summary")
            st.dataframe(summary, use_container_width=True)

            chart_column = st.selectbox("Choose a numeric column for histogram", numeric_columns)
            bins = st.slider("Histogram bins", min_value=5, max_value=100, value=30)
            st.subheader(f"Distribution of {chart_column}")
            st.bar_chart(df[chart_column].dropna().value_counts(bins=bins).sort_index())

            if len(numeric_columns) > 1:
                st.subheader("Pairwise correlation")
                st.dataframe(df[numeric_columns].corr(), use_container_width=True)
        else:
            st.info("No numeric columns detected for charting.")

    st.header("Cleaned dataset preview")
    st.dataframe(cleaned_df.head(10), use_container_width=True)
    st.markdown(
        "Use the controls on the left to adjust the cleaning strategy. The cleaned dataset is available for download below."
    )

    csv_bytes = get_download_data(cleaned_df)
    st.download_button(
        label="Download cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
