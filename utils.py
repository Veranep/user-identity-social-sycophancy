import numpy as np
import pandas as pd

persona_types = [
    "hobbies",
    "motivation",
    "recognition",
    "life_story",
    "crossroads",
    "family",
    "influence",
    "setback",
    "politics",
    "assumptions",
]

wc_persona_types = ["career", "creative"]

feature_cols = {
    "emotions": [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "e_neutral",
    ],
    "liwc": [
        "liwc_Segment",
        "liwc_WC",
        "liwc_Analytic",
        "liwc_Clout",
        "liwc_Authentic",
        "liwc_Tone",
        "liwc_WPS",
        "liwc_BigWords",
        "liwc_Dic",
        "liwc_Linguistic",
        "liwc_function",
        "liwc_pronoun",
        "liwc_ppron",
        "liwc_i",
        "liwc_we",
        "liwc_you",
        "liwc_shehe",
        "liwc_they",
        "liwc_ipron",
        "liwc_det",
        "liwc_article",
        "liwc_number",
        "liwc_prep",
        "liwc_auxverb",
        "liwc_adverb",
        "liwc_conj",
        "liwc_negate",
        "liwc_verb",
        "liwc_adj",
        "liwc_quantity",
        "liwc_Drives",
        "liwc_affiliation",
        "liwc_achieve",
        "liwc_power",
        "liwc_Cognition",
        "liwc_allnone",
        "liwc_cogproc",
        "liwc_insight",
        "liwc_cause",
        "liwc_discrep",
        "liwc_tentat",
        "liwc_certitude",
        "liwc_differ",
        "liwc_memory",
        "liwc_Affect",
        "liwc_tone_pos",
        "liwc_tone_neg",
        "liwc_emotion",
        "liwc_emo_pos",
        "liwc_emo_neg",
        "liwc_emo_anx",
        "liwc_emo_anger",
        "liwc_emo_sad",
        "liwc_swear",
        "liwc_Social",
        "liwc_socbehav",
        "liwc_prosocial",
        "liwc_polite",
        "liwc_conflict",
        "liwc_moral",
        "liwc_comm",
        "liwc_socrefs",
        "liwc_family",
        "liwc_friend",
        "liwc_female",
        "liwc_male",
        "liwc_Culture",
        "liwc_politic",
        "liwc_ethnicity",
        "liwc_tech",
        "liwc_Lifestyle",
        "liwc_leisure",
        "liwc_home",
        "liwc_work",
        "liwc_money",
        "liwc_relig",
        "liwc_Physical",
        "liwc_health",
        "liwc_illness",
        "liwc_wellness",
        "liwc_mental",
        "liwc_substances",
        "liwc_sexual",
        "liwc_food",
        "liwc_death",
        "liwc_need",
        "liwc_want",
        "liwc_acquire",
        "liwc_lack",
        "liwc_fulfill",
        "liwc_fatigue",
        "liwc_reward",
        "liwc_risk",
        "liwc_curiosity",
        "liwc_allure",
        "liwc_Perception",
        "liwc_attention",
        "liwc_motion",
        "liwc_space",
        "liwc_visual",
        "liwc_auditory",
        "liwc_feeling",
        "liwc_time",
        "liwc_focuspast",
        "liwc_focuspresent",
        "liwc_focusfuture",
        "liwc_Conversation",
        "liwc_netspeak",
        "liwc_assent",
        "liwc_nonflu",
        "liwc_filler",
        "liwc_AllPunc",
        "liwc_Period",
        "liwc_Comma",
        "liwc_QMark",
        "liwc_Exclam",
        "liwc_Apostro",
        "liwc_OtherP",
        "liwc_Emoji",
    ],
    "sentiment": ["negative", "s_neutral", "positive"],
    "politeness": ["politeness"],
    "social_dimensions": [
        "social_support",
        "similarity",
        "romance",
        "respect",
        "power",
        "knowledge",
        "identity",
        "fun",
        "conflict",
        "trust",
    ],
}


def extract_demographics(df):
    df["gender"] = df["demographic_information"].apply(
        lambda x: dict(x)["Select Your Gender"] if pd.notnull(x) else np.nan
    )
    df["education"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Select Your Highest Level of Education"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["occupation"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Provide Your Occupation. (_NA if not applicable_)"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["country"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Provide Your Country of Residence."]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["race"] = df["demographic_information"].apply(
        lambda x: dict(x)["Select Your Race"] if pd.notnull(x) else np.nan
    )
    df["nationality"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Provide Your Nationality."] if pd.notnull(x) else np.nan
        )
    )
    df["relationship"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Select Your Relationship Status"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["political"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Select Your Political Orientation"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["income"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Select Your Income Category"] if pd.notnull(x) else np.nan
        )
    )
    df["income_sat"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)["Select Your Income Satisfaction"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["religion"] = df["demographic_information"].apply(
        lambda x: dict(x)["Select Your Religion"] if pd.notnull(x) else np.nan
    )
    df["conditions"] = df["demographic_information"].apply(
        lambda x: (
            dict(x)[
                "Do you have any of the following conditions or difficulties?"
            ]
            if pd.notnull(x)
            else np.nan
        )
    )

    df["culture_pride"] = df["sociodemographic_behavior"].apply(
        lambda x: (
            dict(x)["Do you feel proud of your cultural background?"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["community_connected"] = df["sociodemographic_behavior"].apply(
        lambda x: (
            dict(x)["How connected do you feel to your local community?"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["community_welcome"] = df["sociodemographic_behavior"].apply(
        lambda x: (
            dict(x)["How welcome do you feel in your community?"]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["race_fairness"] = df["sociodemographic_behavior"].apply(
        lambda x: (
            dict(x)[
                "How often do you worry about being treated unfairly because of your race or ethnicity?"
            ]
            if pd.notnull(x)
            else np.nan
        )
    )
    df["culture_skills"] = df["sociodemographic_behavior"].apply(
        lambda x: (
            dict(x)[
                "Have you learned traditional skills, arts, or crafts from your culture?"
            ]
            if pd.notnull(x)
            else np.nan
        )
    )

    df["care"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It’s very important to me to help the people around me. I want to care for others."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["money"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is important to me to be rich. I want to have a lot of money and expensive things."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["equality"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"I think it is important that every person in the world be treated equally. I believes everyone should have equal opportunities in life."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["admiration"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It’s very important to me to show my abilities. I want people to admire what I do."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["humility"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is important to me to be humble and modest. I try not to draw attention to myself."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["spoil"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"Having a good time is important to me. I like to “spoil” myself."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["risks"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"I like to take risks. I always looking for adventures."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["security"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '" It is important to me to live in secure surroundings. I avoid anything that might endanger my safety."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["independence"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"I like to be free and independent. I want to make my own decisions."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["leadership"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is important to me to be in charge and tell others what to do."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["loyalty"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is very important to me to be loyal to my friends. I want to devote myself to people close to me."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["nature"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"I strongly believe that people should care for nature. Looking after the environment is important to me."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["success"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"Being very successful is important to me. I hope people will recognize my achievements."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["customs"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is important to me to follow the customs handed down by my religion or family."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["indulgence"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"Enjoying life’s pleasures is important to me. I like to indulge myself."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["novelty"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"I look for new things to try. I like to do things differently."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["government"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is important to me that the government ensures my safety against all threats."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["proper"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"It is important to me to always behave properly. I want to avoid doing anything people would say is wrong."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["creativity"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"Thinking up new ideas and being creative is important to me. I like to do things in my own original way."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["respect"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                dict(x)[
                    '"I believe everyone should be treated with respect, even people I disagrees with."'
                ]
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_positive_social_impact"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Making a positive social impact"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_start_business"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Starting a business or nonprofit"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_fulfilling_career"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Building a fulfilling career"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_creative_life"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Living a creative life"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_personal_freedom"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Having personal freedom and flexibility"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_family"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Creating or raising a family"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["short-term_goals_expert"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Becoming a thought leader or expert"
                    in dict(x)[
                        "Which of the following best describe your *short-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )

    df["long-term_goals_financial_independence"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Achieving financial independence"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )

    df["long-term_goals_positive_social_impact"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Making a positive social impact"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["long-term_goals_start_business"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Starting a business or nonprofit"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["long-term_goals_fulfilling_career"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Building a fulfilling career"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["long-term_goals_creative_life"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Living a creative life"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["long-term_goals_personal_freedom"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Having personal freedom and flexibility"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["long-term_goals_family"] = (
        df["personal_values_and_motivations"].apply(
            lambda x: (
                1
                * (
                    "Creating or raising a family"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
    ).astype(float)
    df["long-term_goals_expert"] = (
        df["personal_values_and_motivations"]
        .apply(
            lambda x: (
                1
                * (
                    "Becoming a thought leader or expert"
                    in dict(x)[
                        "Which of the following best describe your *long-term* goals?"
                    ]
                )
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )

    df["outgoing"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am outgoing and sociable"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["dominant"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am dominant, and I act as a leader"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["energetic"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am full of energy"'] if pd.notnull(x) else np.nan
            )
        )
        .astype(float)
    )
    df["quiet"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I tend to be quiet "'] if pd.notnull(x) else np.nan
            )
        )
        .astype(float)
    )
    df["follower"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I prefer to have others take charge"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["less_active"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am less active than others "']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["compassionate"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am compassionate. I have a soft heart"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["respectful"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am respectful. I treat others with dignity"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["unsympathetic"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I feel little sympathy for others"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["argumentative"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I like to start arguments with others "']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["rude"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am sometimes rude to others"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["organized"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am organized. I like to keep things in order."']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["careful"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I work carefully to avoid making mistakes"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["messy"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I leave a mess and tend to not clean up"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["careless"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am careless. I make mistakes"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["irresponsible"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"Sometimes I behave irresponsibly"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["worries"] = (
        df["personality_traits"]
        .apply(
            lambda x: dict(x)['"I worry a lot"'] if pd.notnull(x) else np.nan
        )
        .astype(float)
    )
    df["tense"] = (
        df["personality_traits"]
        .apply(
            lambda x: dict(x)['"I can be tense"'] if pd.notnull(x) else np.nan
        )
        .astype(float)
    )
    df["sensitive"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am emotionally sensitive. I get easily upset"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["relaxed"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am relaxed. I handle stress well"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["not_anxious"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I rarely feels anxious or afraid "']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["calm"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I stay calm in difficult situations "']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["original"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am original. I come up with new ideas"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["art_beauty"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I value art and beauty"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["curious"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I am curious about different things"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["not_abstract"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I have little interest in abstract ideas"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["avoid_discussions"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I avoid intellectual and philosophical discussions"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    df["not_artistic"] = (
        df["personality_traits"]
        .apply(
            lambda x: (
                dict(x)['"I have few artistic interests"']
                if pd.notnull(x)
                else np.nan
            )
        )
        .astype(float)
    )
    return df
