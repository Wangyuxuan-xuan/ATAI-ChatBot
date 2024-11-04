SYNONYMS = {
    "director": ["director", "directed", "directs", "direct"],
    "release_date": ["release date", "released"],
    "award": ["award", "oscar", "prize"],
    "production_company": ["production company", "produced"],
    "language": ["language", "original language"],
    "screenwriter": ["screenwriter", "writer"]
}

SPARQL_RELATION_MAPPING = {
    "director": "director",
    "release_date": "publication date",
    "award": "award received",
    "production_company": "production company",
    "language": "original language of film or TV show",
    "screenwriter": "screenwriter"
}

EMBEDDING_REL_MAPPING = {
    "director": ["director", "directed", "directs", "direct"],
    "award": ["award", "oscar", "prize"],
    'publication date': ['release', 'date', 'released', 'releases','release date', 'publication', 'launch', 'broadcast','launched'],
    'executive producer': ['showrunner', 'executive producer'],
    'screenwriter': ['screenwriter', 'scriptwriter', 'writer', 'story'],
    'film editor': ['editor', 'film editor'],
    'box office': ['box', 'office', 'funding', 'box office'],
    'cost': ['budget', 'cost'],
    'nominated for': ['nomination', 'award', 'finalist', 'shortlist', 'selection', 'nominated for'],
    'production company': ['company', 'company of production', "produced", 'production company'],
    'country of origin': ['origin', 'country', 'country of origin'],
    'cast member' :['actor', 'actress', 'cast', 'cast member'],
    'genre': ['type', 'kind', 'genre'],
}

GREETING_SET = {
    "hello", "hello there", "hi there", "hi", "hi hi", "hey", "hi mate", "hey mate", "greetings", "what's up", "good day", "good morning", "good evening", "good afternoon",
    "hey there", "hiya", "morning", "evening", "afternoon", "hallo", "gut morgen", "who are you", "who are u", "how are you", "how are u"
}

# intermediate responses
INITIAL_RESPONSES = [
    "Let me search that information for you.",
    "Checking the details for you.",
    "Let me gather the information for you.",
    "Let me find that out for you :)",
    "Let me look it up.",
    "I'm on it",
    "Checking the data for you.",
    "I'll look into that right away :)",
    "One moment while I check that.",
    "Let me confirm that for you.",
    "I'll dig up the details for you.",
    "On it, just a sec!",
    "Hang tight, I'll get that info.",
    "Let me get that for you.",
]


PERIODIC_RESPONSES = [
    "Still working on it, please hold on...",
    "Give me a moment, I'm working on it...",
    "I'm still gathering the information, one moment...",
    "This is taking a bit longer, please be patient...",
    "I'm on it, still fetching the details...",
    "Please hold on, I'm processing your request...",
    "It might take a few more seconds, please wait...",
    "Still checking, almost there...",
    "Working on it, thank you for your patience...",
    "This will take just a little more time...",
    "Hang tight, I'm finding the answer...",
    "I'm getting the answer, please hold on...",
    "Almost done, just another moment...",
    "Hold on, I'm looking it up...",
]
