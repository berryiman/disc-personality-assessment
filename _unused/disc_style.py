import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import StringIO
import math


st.set_page_config(
    page_title="DISC Personality Assessment :bust_in_silhouette:",
    layout="wide",
    page_icon=":bust_in_silhouette:",
)

# Custom CSS to improve app appearance
st.markdown(
    """
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #184b6a;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background-color: #d3d3d3d3;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App Title with custom styling
st.markdown(
    "<h1 style='text-align: center; color: #184b6a;'>DISC Personality Assessment 👤</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-style: italic;'>Discover your DISC personality style by answering the questions below.</p>",
    unsafe_allow_html=True,
)

# Ensure session state variables are initialized
if "started" not in st.session_state:
    st.session_state.started = False

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "raw_score" not in st.session_state:
    st.session_state.raw_score = {"D": 0, "I": 0, "S": 0, "C": 0}

if "score" not in st.session_state:
    st.session_state.score = {"D": 0, "I": 0, "S": 0, "C": 0}

# If the user hasn't started the assessment yet
if not st.session_state.started:
    st.markdown(
        """
    ### Welcome to the DISC Personality Assessment

    The DISC assessment is a tool that helps you understand your personality traits across four major dimensions: Dominance (D), Influence (I), Steadiness (S), and Conscientiousness (C). By answering a series of questions, you'll discover your personality style and gain insights into how you interact with others.

    #### Instructions:
    - You will be presented with a series of statements.
    - For each statement, indicate how much you agree or disagree using the options provided.
    - A value of **1** completely disagree, **2** somehow disagree, **3** neutral, **4** somehow agree, **5** completely agree
    - Once you complete the assessment, you'll receive your DISC style profile and a detailed breakdown of your results.
    - Carefully read the descriptions as some of them sound similar but have different meanings.

    Click "Let's Begin" to start the assessment!
    """
    )

    # Create layout for buttons
    c1, c2, c3 = st.columns([1, 2, 1])

    # Handle the "Let's Begin" button press
    if c1.button("Let's Begin"):
        st.session_state.started = True
        st.session_state.submitted = False
        st.rerun()  # Rerun the script to move to the next stage

    # File upload option, which is shown after clicking "Upload Previous Results"
    if c3.checkbox("Upload Previous Results"):
        # Display the file uploader when the button is clicked
        uploaded_file = st.file_uploader(
            "Upload your previous JSON results", type=["json"]
        )
        if uploaded_file is not None:
            # Read the file as bytes
            bytes_data = uploaded_file.getvalue()

            # Convert bytes to string and then to a StringIO object
            stringio = StringIO(bytes_data.decode("utf-8"))

            # Read the JSON content from the string
            try:
                # Load the JSON content
                uploaded_file_content = json.load(stringio)
                # Assume the uploaded content is the normalized_score
                st.session_state.normalized_score = uploaded_file_content

                # Set session states to move forward
                st.session_state.started = True
                st.session_state.show_results = True
                st.session_state.submitted = True
                st.write("File uploaded and processed successfully!")
                st.rerun()  # Rerun to process the results

            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON: {e}")


# Updated plot function
def create_disc_plot(resultant_angle, resultant_magnitude):
    # Reduce the size of the figure by modifying figsize
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": "polar"}
    )  # Reduce size from 10x10 to 6x6

    # The rest of your plotting code remains unchanged
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.01)

    ax.plot(
        resultant_angle,
        resultant_magnitude,
        "o",
        markersize=24,
        color="#4CAF50",
        label="Your DISC Style",
    )

    categories = ["D", "I", "S", "C"]
    angles = [7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4]
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=14, fontweight="bold")

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=np.pi, color="gray", linestyle="--", alpha=0.7)

    ax.axvline(x=np.pi / 2, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=3 * np.pi / 2, color="gray", linestyle="--", alpha=0.7)

    ax.set_yticklabels([])

    ax.grid(True, alpha=0.3)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#f0f2f6")

    plt.title("Your DISC Style Profile", fontsize=16, fontweight="bold", pad=20)

    return fig


# Function to download JSON data
def get_json_download_link(normalized_score):
    json_str = json.dumps(normalized_score, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="disc_results.json">Download JSON Results</a>'
    return href


def get_json_download_button(normalized_score):
    # Convert the normalized score dictionary into a JSON string
    json_str = json.dumps(normalized_score, indent=2)

    # Create a downloadable button using the JSON data
    st.download_button(
        label="Download JSON Results",
        data=json_str,
        file_name="disc_results.json",
        mime="application/json",
    )


# Function to download PDF report
def get_pdf_download_link(pdf_buffer):
    # Encode the PDF buffer in base64 for downloading
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="disc_report.pdf">Download PDF Report</a>'
    return href


def get_pdf_download_button(pdf_buffer):
    # Encode the PDF buffer in base64
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()

    # Convert base64 PDF into a downloadable button
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer.getvalue(),
        file_name="disc_report.pdf",
        mime="application/pdf",
    )


def describe_style(normalized_score, resultant_angle):
    """
    Determine the user's DISC style based on the resultant vector (angle and magnitude) on the DISC wheel.
    Args:
        normalized_score: Dictionary of normalized scores for D, I, S, C styles.
        resultant_angle: The resultant angle in radians.
        resultant_magnitude: The resultant magnitude (how strongly the traits are combined).

    Returns:
        A string description of the DISC style.
    """

    # Convert the resultant angle from radians to degrees
    resultant_degrees = math.degrees(resultant_angle)
    if resultant_degrees < 0:
        resultant_degrees += 360  # Convert negative angles to positive

    # Define the angular ranges for main styles and combinations
    style_ranges = {
        # D (Dominance)
        "D": (315, 337.5),
        "DC": (270, 315),
        "DI": (337.5, 360),  # Also covers the 0 degree point
        
        # I (Influence)
        "I": (45, 67.5),
        "ID": (0, 45),
        "IS": (67.5, 90),
        
        # S (Steadiness)
        "S": (135, 157.5),
        "SI": (90, 135),
        "SC": (157.5, 180),
        
        # C (Conscientiousness)
        "C": (225, 247.5),
        "CS": (180, 225),
        "CD": (247.5, 270)
    }

    # Check if all normalized scores are equal (balanced style)
    if all(score == list(normalized_score.values())[0] for score in normalized_score.values()):
        st.markdown("### Balanced Style")
        st.markdown(
            "Your responses indicate a balanced personality, where you do not show a clear preference for any specific DISC style."
        )
        return "Balanced Style"

    # Determine which range the resultant angle falls into
    for style, (start_angle, end_angle) in style_ranges.items():
        if start_angle <= resultant_degrees < end_angle or (start_angle == 337.5 and resultant_degrees == 0):
            # If it's a combination style, look up the combination description
            description = disc_descriptions["single"][style]
            
            # Display the result
            st.markdown(f"{description['title']}\n\n{description['description']}")
            st.markdown(f"**Strengths:** {description['strengths']}")
            st.markdown(f"**Challenges:** {description['challenges']}")
            return f"{description['title']}\n\n{description['description']}\n\nStrengths: {description['strengths']}\n\nChallenges: {description['challenges']}"
    
    # Default fallback if no match is found
    st.markdown("### Balanced Style")
    st.markdown(
        "Your responses indicate a balanced personality without a clear preference for any specific DISC style."
    )
    return "Balanced Style"



def normalize_scores(scores, questions):
    max_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}
    min_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}

    for q in questions:
        for style in ["D", "I", "S", "C"]:
            mapping = q["mapping"][style]
            if mapping >= 0:
                max_contribution = mapping * 2  # Max when (answer - 3) = +2
                min_contribution = mapping * (-2)  # Min when (answer - 3) = -2
            else:
                max_contribution = mapping * (-2)  # Max when (answer - 3) = -2
                min_contribution = mapping * 2  # Min when (answer - 3) = +2

            max_possible_scores[style] += max_contribution
            min_possible_scores[style] += min_contribution

    print(f"Max possible scores: {max_possible_scores}")
    print(f"Min possible scores: {min_possible_scores}")

    normalized_scores = {}
    for style in ["D", "I", "S", "C"]:
        # Ensure the raw score is within the possible range
        score = max(min(scores[style], max_possible_scores[style]), min_possible_scores[style])
        score_range = max_possible_scores[style] - min_possible_scores[style]
        if score_range == 0:
            normalized_scores[style] = 50.0  # Neutral score if no variation is possible
        else:
            normalized_scores[style] = ((score - min_possible_scores[style]) / score_range) * 100
            # Ensure the normalized score is within 0 to 100
            normalized_scores[style] = max(0, min(normalized_scores[style], 100))
    return normalized_scores


# Function to create PDF report
def create_pdf_report(normalized_score, relative_percentages, fig, style_description):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    # Define custom styles for better formatting
    styles.add(ParagraphStyle(name='Justify', alignment=4, leading=12))
    styles.add(ParagraphStyle(name='Heading2Center', parent=styles['Heading2'], alignment=1))
    styles.add(ParagraphStyle(name='BodyTextCenter', parent=styles['BodyText'], alignment=1))
    
    story = []

    # Title
    story.append(Paragraph("DISC Personality Assessment Report", styles["Title"]))
    story.append(Spacer(1, 20))

    # Add Introduction
    story.append(Paragraph("Thank you for completing the DISC Personality Assessment. This report provides insights into your personality style based on your responses.", styles['Justify']))
    story.append(Spacer(1, 20))

    # Add DISC Style Breakdown (Absolute Scores)
    story.append(Paragraph("Your DISC Style Breakdown (Absolute Scores):", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("The following scores represent your absolute level in each DISC style on a scale from 0% to 100%. A higher percentage indicates a stronger tendency towards that style.", styles['Justify']))
    story.append(Spacer(1, 10))

    # Create a table for absolute scores
    data = [['Style', 'Score (0-100%)']]
    for style, score in normalized_score.items():
        data.append([style, f"{score:.2f}%"])

    table = Table(data, hAlign='LEFT', colWidths=[100, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Add DISC Style Breakdown (Relative Percentages)
    story.append(Paragraph("Your DISC Style Breakdown (Relative Percentages):", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("These percentages represent the proportion of each DISC style relative to your overall personality profile. The total sums up to 100%.", styles['Justify']))
    story.append(Spacer(1, 10))

    # Create a table for relative percentages
    data = [['Style', 'Relative Percentage']]
    for style, score in relative_percentages.items():
        data.append([style, f"{score:.2f}%"])

    table = Table(data, hAlign='LEFT', colWidths=[100, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Explanation about the difference between Absolute Scores and Relative Percentages
    story.append(Paragraph("Understanding Your Scores:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The absolute scores indicate how strongly you exhibit each DISC style on its own, without comparison to other styles. A higher score means you tend to display more behaviors associated with that style.\n\n"
        "The relative percentages show how each style contributes to your overall personality profile compared to the other styles. These percentages sum up to 100% and help you understand which styles are most dominant in your personality.",
        styles['Justify']
    ))
    story.append(Spacer(1, 100))

    # Add the plot as an image
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
    img_buffer.seek(0)
    img = Image(img_buffer, width=400, height=400)
    story.append(Spacer(1, 10))
    story.append(img)
    story.append(Spacer(1, 20))

    # Add personalized style description
    story.append(Paragraph("Your Personalized DISC Style Description:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(style_description.replace("###", ""), styles['Justify']))
    story.append(Spacer(1, 50))

    story.append(PageBreak())
    # Add explanation about each DISC style
    story.append(Paragraph("Understanding All DISC Styles:", styles["Heading2"]))
    story.append(Spacer(1, 10))
    styles_list = [
        ('Dominance (D):', 'You tend to be direct, results-oriented, and assertive. You are motivated by challenges and achieving tangible results.'),
        ('Influence (I):', 'You are typically outgoing, enthusiastic, and optimistic. You enjoy social interactions and persuading others.'),
        ('Steadiness (S):', 'You are often patient, supportive, and team-oriented. You value cooperation and harmony in relationships.'),
        ('Conscientiousness (C):', 'You tend to be analytical, precise, and detail-oriented. You focus on accuracy, quality, and expertise.')
    ]
    for title, description in styles_list:
        story.append(Paragraph(f"<b>{title}</b> {description}", styles['Justify']))
        story.append(Spacer(1, 5))
    story.append(Spacer(1, 10))

    # Add final remarks
    story.append(
        Paragraph(
            "Remember, everyone has aspects of all four styles, but most people tend to gravitate towards one or two primary styles. "
            "Your unique combination of styles influences how you communicate, make decisions, and interact with others.",
            styles['Justify']
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Use this insight to enhance your personal and professional relationships by recognizing and appreciating different styles in yourself and others.",
            styles['Justify']
        )
    )

    # Build the PDF
    doc.build(story)

    # Ensure we return the buffer to be used for downloading
    buffer.seek(0)
    return buffer


# If the user has started the test, proceed with the questions
if st.session_state.started:

    # Initialize session state variables for the questions
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0

    if "score" not in st.session_state:
        st.session_state.score = {"D": 0, "I": 0, "S": 0, "C": 0}

    if "answers" not in st.session_state:
        st.session_state.answers = {}

    if "show_results" not in st.session_state:
        st.session_state.show_results = False

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if "questions" not in st.session_state:
        questions = json.load(open("questions.json", "r"))
        random.shuffle(questions)
        st.session_state.questions = questions[:30]  # Use the first 30 questions

    questions_per_page = 1  # Show one question at a time
    total_questions = len(st.session_state.questions)
    total_pages = (
        total_questions + questions_per_page - 1
    ) // questions_per_page  # Ceiling division

    # Load DISC descriptions
    disc_descriptions = json.load(open("disc_descriptions.json", "r"))

    if not st.session_state.show_results:
        start = st.session_state.page_number * questions_per_page
        end = start + questions_per_page

        # Calculate progress
        progress = (st.session_state.page_number) / total_questions

        # Display progress bar outside the form
        st.progress(progress)
        
        with st.form(key=f"form_{st.session_state.page_number}"):
            i = start
            if i < total_questions:
                q = st.session_state.questions[i]
                n = i + 1
                st.markdown(f"#### {n}) {q['question']}")
                options = [
                    "Select an option",
                    "1 - Completely Disagree",
                    "2 - Somehow Disagree",
                    "3 - Neutral",
                    "4 - Somehow Agree",
                    "5 - Completely Agree",
                ]
                selected_option = st.radio(
                    "Choose your response",
                    options=options,
                    index=0,
                    key=f"radio_{i}",
                    horizontal=True,
                )
                if st.session_state.page_number < total_pages - 1:
                    submit_button = st.form_submit_button("Next")
                else:
                    submit_button = st.form_submit_button("**Show My DISC Style**")
            else:
                submit_button = st.form_submit_button("**Show My DISC Style**")

        if submit_button:
            if selected_option == "Select an option":
                st.warning("Please select a response to proceed.")
            else:
                # Map the selected option to a score
                score_mapping = {
                    "1 - Completely Disagree": 1,
                    "2 - Somehow Disagree": 2,
                    "3 - Neutral": 3,
                    "4 - Somehow Agree": 4,
                    "5 - Completely Agree": 5,
                }
                st.session_state.answers[i] = score_mapping[selected_option]
                if st.session_state.page_number < total_pages - 1:
                    st.session_state.page_number += 1
                    st.rerun()
                else:
                    # Set flags to show results and indicate submission
                    st.session_state.show_results = True
                    st.session_state.submitted = False  # Ensure this is reset
                    st.rerun()
    else:
        # After the user has completed the assessment or uploaded results
        if not st.session_state.submitted:
            # Reset the scores before calculating
            st.session_state.score = {"D": 0, "I": 0, "S": 0, "C": 0}
            # Calculate raw scores
            for i in range(total_questions):
                q = st.session_state.questions[i]
                answer = st.session_state.answers[i]
                for style in ["D", "I", "S", "C"]:
                    st.session_state.score[style] += q["mapping"][style] * (answer - 3)
            print(f'Raw score: {st.session_state.score}')
            st.session_state.raw_score = st.session_state.score.copy()

            # Normalize the scores
            normalized_score = normalize_scores(st.session_state.score, st.session_state.questions)
            print(f'Normalized score: {normalized_score}')
            st.session_state.normalized_score = normalized_score
            st.session_state.submitted = True  # Set to True to avoid recalculation
        
        
        else:
            # Check if normalized_score exists in session_state
            if 'normalized_score' in st.session_state:
                normalized_score = st.session_state.normalized_score
            else:
                # If not, recalculate it from st.session_state.score
                normalized_score = normalize_scores(st.session_state.score, st.session_state.questions)
                st.session_state.normalized_score = normalized_score

        print(f'Normalized score: {normalized_score}')
        
        # Define the categories and their positions
        categories = ["D", "I", "S", "C"]

        # Angles for the styles
        angles = [7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4]

        # Prepare the values
        values = [normalized_score[cat] for cat in categories]
        
        # Divide Each Normalized Score by 100:
        scaled_scores = {style: score / 100 for style, score in normalized_score.items()}

        # Compute x and y components of the style vectors
        x_components = []
        y_components = []
        for style in categories:
            angle = angles[categories.index(style)]
            magnitude = scaled_scores[style]
            x_components.append(magnitude * np.cos(angle))
            y_components.append(magnitude * np.sin(angle))

        # Sum the components
        total_x = sum(x_components)
        total_y = sum(y_components)

        # Compute the resultant vector
        resultant_magnitude = np.sqrt(total_x**2 + total_y**2)
        resultant_angle = np.arctan2(total_y, total_x)
        
        print(f"Resultant magnitude: {resultant_magnitude}")

        # Ensure the magnitude does not exceed 1
        # resultant_magnitude = min(resultant_magnitude, 1.0)

        # Create the updated plot
        fig = create_disc_plot(resultant_angle, resultant_magnitude)

        # Use Streamlit columns to control the figure width
        col1, col2, col3 = st.columns(
            [1, 2, 1]
        )  # Adjust the middle column width (2/4 of the page width)

        with col2:  # Display the plot in the middle column
            st.pyplot(fig)

        # Personalized Style Descriptions
        st.markdown("## Your Personalized DISC Style")
        style_description = describe_style(normalized_score, resultant_angle)

        # Display normalized scores with progress bars
        
        ### Here is was using the style usage rather than general style breakdown
        # st.markdown("## Your DISC Style Breakdown")
        # st.markdown("### Style Usage Scores (How much you use each style)")
        # cols = st.columns(4)
        # for idx, (style, score_value) in enumerate(normalized_score.items()):
        #     with cols[idx]:
        #         st.markdown(f"**{style}**")
        #         # Ensure score_value is within 0 to 100
        #         score_value = max(0, min(score_value, 100))
        #         # Adjust the progress bar value to be between 0.0 and 1.0
        #         st.progress(score_value / 100)
        #         # Display the score as a percentage
        #         st.text(f"{score_value:.2f}%")
                
        total_normalized = sum(normalized_score.values())
        relative_percentages = {}
        for style, score in normalized_score.items():
            if total_normalized == 0:
                relative_percentages[style] = 0
            else:
                relative_percentages[style] = (score / total_normalized) * 100
        
        st.markdown("## Your DISC Style Breakdown")
        st.write("Relative Percentages")
        cols = st.columns(4)
        for idx, (style, score_value) in enumerate(relative_percentages.items()):
            with cols[idx]:
                st.markdown(f"**{style}**")
                # Ensure score_value is within 0 to 100
                score_value = max(0, min(score_value, 100))
                # Adjust the progress bar value to be between 0.0 and 1.0
                st.progress(score_value / 100)
                # Display the score as a percentage
                st.text(f"{score_value:.2f}%")
        
        # Download options
        st.markdown("## Download Your Results")
        col1, col2 = st.columns(2)
        with col1:
            get_json_download_button(normalized_score)
        with col2:
            pdf_buffer = create_pdf_report(
                            normalized_score=normalized_score,
                            relative_percentages=relative_percentages,
                            fig=fig,
                            style_description=style_description
                        )
            get_pdf_download_button(pdf_buffer)

        # Explanation about DISC styles
        st.markdown("""---""")
        st.markdown(
            """
        ### Understanding All DISC Styles

        - **Dominance (D)**: You tend to be direct, results-oriented, and assertive.
        - **Influence (I)**: You are typically outgoing, enthusiastic, and optimistic.
        - **Steadiness (S)**: You are often patient, supportive, and team-oriented.
        - **Conscientiousness (C)**: You tend to be analytical, precise, and detail-oriented.

        Remember, everyone has aspects of all four styles, but most people tend to gravitate towards one or two primary styles. 
        Your unique combination of styles influences how you communicate, make decisions, and interact with others.
        """
        )

        if st.button("Restart"):
            st.session_state.pop("page_number")
            st.session_state.pop("score")
            st.session_state.pop("answers")
            st.session_state.pop("show_results")
            st.session_state.pop("questions")
            st.rerun()


st.markdown(
    """
    ---
    <div style="text-align: center;">
        <strong><a href="https://github.com/dzyla/disc-personality-assessment">Source code</a></strong> | Developed by <a href="https://dzyla.com">Dawid Zyla</a>
    </div>
    """,
    unsafe_allow_html=True,
)
