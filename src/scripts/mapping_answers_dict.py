
# Dictionary for the columns in the survey
column_mappings = {
    "CR1": "Age",
    "CR2": "Gender",
    "CR8": "Smoke",
    "OR45": "SmokingParents",
    "OR46": "SmokingFriends",
    "OR1": "WorkingParents",
    "CR22": "SeenSmokerInSchool",
    "CR21": "SeenSmokerInPublicPlace",
    "CR20": "SeenSmokerInEnclosedPlace",
    "CR19": "SeenSmokerInHome",
    "OR55": "ParentWarnings",
    "OR49": "AttractiveSmoker",
    "CR41": "HardQuitSmoke",
    "CR42": "SmokerConfidentInCelebrations",
    "CR33": "SchoolWarnings",
    "CR32": "SeenHealthWarnings",
    "CR31": "AntiTobaccoInEvents",
    "CR30": "AntiTobaccoInMedia",
    "CR25": "BanTobaccoOutdoors",
    "CR23": "HarmfulPassiveSmoke"
}

# Dictionary for the questions in the survey
# How old are you?
CR1_dict = {1: "11 years old or younger", 2: "12 years old", 3: "13 years old",
            4: "14 years old", 5: "15 years old", 6: "16 years old", 7: "17 years old or older"}

# What is your sex?
CR2_dict = {1: "Male", 2: "Female"}

# Please think about the days you smoked cigarettes during the past 30 days (one month). How many cigarettes did you usually smoke per day?
CR8_dict = {1: False, 2: True, 3: True,
            4: True, 5: True, 6: True, 7: True}
# CR8_dict = {1: "0", 2: "Less than 1", 3: "1", 4: "2 to 5",
#             5: "6 to 10", 6: "11 to 20", 7: "More than 20"}

# Do your parents smoke tobacco?
OR45_dict = {1: "None", 2: "Both", 3: "Father only",
             4: "Mother only", 5: "Don't know"}

# Do any of your closest friends smoke tobacco?
OR46_dict = {1: "None of them", 2: "Some of them",
             3: "Most of them", 4: "All of them"}

# Do your parents work?
OR1_dict = {1: "Father only", 2: "Mother only",
            3: "Both", 4: "Neither", 5: "Don't know"}

# During the past 30 days, did you see anyone smoke inside the school building or outside on the school property?
CR22_dict = {1: True, 2: False}

# During the past 7 days, on how many days has anyone smoked in your presence, at outdoor public places (playgrounds, sidewalks, entrances to buildings, parks, beaches, swimming pools)?
CR21_dict = {1: "0 days", 2: "1 to 2 days",
             3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}

# During the past 7 days, on how many days has anyone smoked in your presence, inside any enclosed public place, other than your home (such as shops, restaurants, shopping malls, movie theaters)?
CR20_dict = {1: "0 days", 2: "1 to 2 days",
             3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}

# During the past 7 days, on how many days has anyone smoked in your presence, inside your home
CR19_dict = {1: "0 days", 2: "1 to 2 days",
             3: "3 to 4 days", 4: "5 to 6 days", 5: "7 days"}

# Have you ever tried smoking cigarettes?
CR5_dict = {1: "Yes", 2: "No"}

# How old were you when you first tried smoking?
CR6_dict = {1: "Never tried", 2: "7 years old or younger", 3: "8 or 9 years old",
            4: "10 or 11 years old", 5: "12 or 13 years old", 6: "14 or 15 years old", 7: "16 years old or older"}

# Has anyone in your family discussed the harmful effects of smoking tobacco with you?
OR55_dict = {1: True, 2: False}

# Do you think smoking tobacco makes young people look more or less attractive?
OR49_dict = {1: "More attractive", 2: "Less attractive", 3: "No difference"}

# Once someone has started smoking tobacco, do you think it would be difficult for them to quit?
CR41_dict = {1: "Definitely not", 2: "Probably not", 3: "Probably yes", 4: "Definitely yes"}

# Do you think smoking tobacco helps people feel more comfortable or less comfortable at celebrations, parties, or in other social gatherings?
CR42_dict = {1: "More comfortable", 2: "Less comfortable", 3: "No difference"}

# During the past 12 months (1 year) and during lessons, were you taught about the dangers of tobacco use?
CR33_dict = {1: "Yes", 2: "No", 3: "I don't know"}

# During the past 30 days (one month), did you see any health warnings on cigarette packages?
CR32_dict = {1: "Yes, but I didn't think much of them", 2: "Yes, they made me consider quitting or avoiding smoking", 3: "No"}

# During the past 30 days (one month), did you see or hear any anti-tobacco media messages on television, radio, internet, billboards, posters, newspapers, magazines, or movies?
CR30_dict = {1: True, 2: False}

# During the past 30 days (one month), did you see or hear any anti-tobacco messages at sports events, fairs, concerts, or community events, or social gatherings?
CR31_dict = {1: "I did not go to events in the past 30 days", 2: "Yes", 3: "No"}

# Are you in favor of banning smoking at outdoor public places (such as playgrounds, parks, beaches, entrance to buildings, stadium, bus stops)?
CR25_dict = {1: True, 2: False}

# Do you think the smoke from other people's tobacco smoking is harmful to you?
CR23_dict = {1: "Definitely not", 2: "Probably not", 3: "Probably yes", 4: "Definitely yes"}
