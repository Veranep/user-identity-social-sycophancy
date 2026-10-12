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

wc_persona_types = ["career", "creative", "random"]


def get_persona_convo(string_convo, wc=False):
    if wc:
        return string_convo
    string_convo = (
        string_convo.strip()
        .replace("\n", " ")
        .replace('12"x12"', "12x12 inches")
        .replace("\u00d7", "x")
        .replace('1/2"', "1/2 inch")
        .replace('1"x2"', "1x2 inches")
        .replace('3/4"', "3/4 inch")
        .replace('24"x8"', "24 x 8 inches")
        .replace('4"', "4 inch")
        .replace('12" x 8"', "12 x 8 inches")
        .replace('"assistant":', '"assistant",')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
        .replace('outcome‑focused): \\"I', "outcome‑focused): 'I")
        .replace('design.\\" \\n\\nE', "design.' \\n\\nE")
        .replace('skills): \\"I', "skills): 'I")
        .replace('experience.\\" \\n\\nE', "experience.' \\n\\nE")
        .replace('confident): \\"I combine', "confident): 'I combine")
        .replace('projects.\\" \\n\\nOne', "projects.' \\n\\nOne")
        .replace('tagline: \\"Artist', "tagline: 'Artist")
        .replace(
            'experiences.\\" \\n\\nLinkedIn',
            "experiences.' \\n\\nLinkedIn",
        )
        .replace('paragraphs): \\"I', "paragraphs): 'I")
        .replace('tech." \\n\\nDelivery', "tech.' \\n\\nDelivery")
        .replace('more." This', "more.” This")
        .split("  // Note:")[0]
    )
    if string_convo[-4:] == "]} ]":
        string_convo = string_convo[:-4] + "}]"
    elif string_convo[-4:] == "   ]":
        string_convo = string_convo[:-4] + "}]"
    if string_convo[-3:] != '"}]':
        if string_convo[-1] == ".":
            string_convo += '"}]'
        elif string_convo[-3:] == '"]}':
            string_convo = string_convo[:-3] + '"}]'
        elif string_convo[-3:] == "} ]":
            string_convo = string_convo[:-3] + "}]"
        elif string_convo[-3:] == "]}]":
            string_convo = string_convo[:-3] + "}]"
        elif string_convo[-3:] == "}]}":
            string_convo = string_convo[:-3] + "}]"
        elif string_convo[-3:] == "}}]":
            string_convo = string_convo[:-3] + "}]"
        if string_convo[-3:] == ".}]":
            string_convo = string_convo[:-3] + '."}]'
        elif string_convo[-3:] == ".]}":
            string_convo = string_convo[:-3] + '."}]'
    persona_convo = eval(string_convo)
    mod = 0
    for t in range(len(persona_convo)):
        if (
            len(persona_convo) > 2
            and persona_convo[t - mod]["role"] == "assistant"
            and persona_convo[t - 1 - mod]["role"] == "assistant"
        ):
            persona_convo[t - 1 - mod]["content"] = (
                persona_convo[t - 1 - mod]["content"]
                + " "
                + persona_convo[t - mod]["content"]
            )
            del persona_convo[t - mod]
            mod += 1
    return persona_convo


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
