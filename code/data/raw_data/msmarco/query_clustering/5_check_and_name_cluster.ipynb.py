{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import random\n",
    "import json\n",
    "\n",
    "os.chdir('/shared/eng/pj20/lc/DeepRetrieval')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('outputs/clustered_queries.json', 'r') as f:\n",
    "    data = json.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "sys.path.append(\"./\")\n",
    "from azure_api import azure_client\n",
    "\n",
    "\n",
    "def llm_name_cluster(queries):\n",
    "    prompt = f\"\"\"\n",
    "You are a helpful assistant that give the topic of clusters of queries.\n",
    "Here are the queries:\n",
    "{queries}\n",
    "Please provide a single and general word that represents the main topic of this cluster.\n",
    "Your response should be without any additional explanations.\n",
    "\"\"\"\n",
    "    response = azure_client.chat.completions.create(\n",
    "        model=\"gpt-4o\",\n",
    "        messages=[{\"role\": \"user\", \"content\": prompt}],\n",
    "    ).choices[0].message.content\n",
    "\n",
    "    return response\n",
    "\n",
    "\n",
    "def llm_name_all_cluster(query_list):\n",
    "    prompt = f\"\"\"\n",
    "You are a helpful assistant who provides the topics for a list of query clusters.\n",
    "Here are the query clusters:\n",
    "{query_list}\n",
    "Please provide a single, general word that represents the main topic of each cluster.\n",
    "The topic name should be concise and informative.\n",
    "Your response should contain only the topic words, without any additional explanations, like:\n",
    "0. topic word A\n",
    "1. topic word B\n",
    "2. topic word C\n",
    "...\n",
    "\"\"\"\n",
    "    response = azure_client.chat.completions.create(\n",
    "        model=\"gpt-4o\",\n",
    "        messages=[{\"role\": \"user\", \"content\": prompt}],\n",
    "    ).choices[0].message.content\n",
    "\n",
    "    return response\n",
    "\n",
    "\n",
    "\n",
    "# print(llm_name_cluster('test'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "peek_size = 10\n",
    "random.seed(42)\n",
    "\n",
    "\n",
    "query_list = {}\n",
    "\n",
    "for key, queries in data.items():\n",
    "    sampled_queries = random.sample(queries, peek_size)\n",
    "    query_list[key] = sampled_queries\n",
    "\n",
    "query_list = dict(sorted(query_list.items(), key=lambda item: int(item[0])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cluster id: 0, queries: ['how big is greenwood lake', 'where do cruise ships ibiza spain', 'where is network place in chicago?', 'what day is halloween', 'when do leaves change in birmingham al', 'how long is the atlantic city nj boardwalk', 'when is the best time to harvest plantain seeds?', 'how long has michigan adventure been open', 'what continent is cairo in', 'what is the best time of year to visit paris']\n",
      "cluster id: 1, queries: ['what county is la mirada in', 'what county is plant city', 'what county is long beach cain', 'what district is rex mill middle school in', 'harvey il what county', 'vincent ohio what county is that', 'what county is breckenridge co?', 'what county is nunica mi in', 'what county is hinckley maine', 'what parish is cottonport, la. in']\n",
      "cluster id: 2, queries: ['what is synthesized on rer', 'how do you test a substance for acid or alkali', 'how many lines of symmetry are there in a regular trapezium', 'what type of radiation can humans see', 'what do pelican eels feed on?', 'what different kind of fish can breed together', 'describe how a nuclear reactor works', 'what two amino acids are used in the food industry as nutritional supplements in bread products', 'difference between simple diffusion and active transport', 'what is the relevance of observing the osmotic properties of the potato cell/ plant cell?']\n",
      "cluster id: 3, queries: ['How long does it take to get a copy of a lost car title in kansas', 'how long dog panting heavy after exercise', 'how long does it take to become a nurses aide', 'how long to bake a 11x15x2 cake', 'how long after intercourse can someone be pregnant?', 'who long is the surgery recovery time for endometriosis surgery?', 'how long to bake small spaghetti squash', 'how long does a jar of coffee last', 'how long can you keep strawberries in the fridge', 'how long is the recovery time for umbilical hernia belly button repair']\n",
      "cluster id: 4, queries: ['what is cross collateralization', 'what is base in computers', 'what is bsides conference', 'what is the culture and society in australia', 'what is clearquest', 'what are some the strategies developed by airmen within the air force?', 'what is the difference between civil service and federal employee', 'what does cm@risk mean', 'what is the kula foundation', 'what is mediation before a trial']\n",
      "cluster id: 5, queries: ['how was the treaty of versailles a trigger to wwii', 'how did the solidarity movement led to the collapse of communism in poland', 'what political party is north korea', 'when did hammurabi die', 'what parties were involved in the tinker vs des moines case', 'how napoleon bonaparte reorganized education', 'how did the tea party start', 'what is hillary clinton occupation or cause', 'who is a liberal?', 'why did monroe make a good president']\n",
      "cluster id: 6, queries: ['whole foods list', 'most important mineral for the body', 'calories in kfc chicken drumstick and thigh original recipe', \"what neurotransmitter makes you crave something you previously didn't want\", 'food that are vegetarian that have vitamin a', 'how many calories should i burn in an hour water aerobics class', 'what are the healthy benefits from eating hard boiled eggs', 'what some nutritional complication during pregnancy', 'how does water help with digestion', 'how much protein is in an average cube steak']\n",
      "cluster id: 7, queries: ['what is similar to triclopyr', 'what is the prefixes of five', 'What Does FLB Stand for', 'what is schnapps made from', 'what is cordierite stone', 'what is the eye plant called?', \"what's mine is yours saying\", \"what's a transgender man\", 'what does the dream mean of running to catch the a bus', 'what is dinner dressing in great expectations']\n",
      "cluster id: 8, queries: ['can lyme disease cause a runny nose', 'can your chest hurt from allergies', 'why is my breath sometimes hot and sometimes cold', 'do you need glasses after lasik', 'can fish cause gout attack', 'what causes swelling of left ankle only', 'awesome are symptoms scratching', 'at what age do kittens get their first shots', 'what helps bring blood pressure up', 'does high blood pressure make you feel tired and have no energy?']\n",
      "cluster id: 9, queries: ['authorization on credit card meaning', 'phone number american airlines info', 'vt transaction plus customer number crack', 'safelink toll free number', 'bauer hockey phone number', 'bentonville police non emergency number', 'quest diagnostics bill pay phone number', 'straight talk refill phone number', 'suntrust payoff phone number auto loan', 'arn transaction number']\n",
      "cluster id: 10, queries: ['how to make crepe myrtles bloom', 'how to find a judgement lien', 'how airplanes make taxi', 'how to bring resting heart rate down', 'how to format excel cells based on size', 'how to find out where a sql view is being used', 'how to cash substitution check', 'how to creating milestones in sharepoint', 'how to add photo on instagram on laptop', 'toad how to use sql loader to load xml']\n",
      "cluster id: 11, queries: ['is there a music app that play offline', 'what are arf files', 'what app do i use for jennov camera', \"what is dvd drive's\", 'how do you open navicure rpt files', 'what cable company is suddenlink under?', 'where to find my oracle password file 11g', 'what operating system does an iphone have', 'what network is orange is the new black on', 'can I filter location history by device?']\n",
      "cluster id: 12, queries: ['causes of avulsion fracture', 'is medical marijuana approved by the fda', 'symptoms of hemothorax', 'which cells are attacked in rheumatoid arthritis', 'side effects of thyroid medication', 'cause of oily skin and hair loss', 'where is pectineus muscle', 'medication for gout im injection', \"is peyronie's disease inherited\", 'serum urea level normal range']\n",
      "cluster id: 13, queries: ['iso acronym meaning for computer applications', 'is rooted traduccion', 'is the merrymeeting river a a river or lake', 'irish names for girls and their meanings', 'weber surname meaning', 'werther meaning', 'meaning if a climate is temperate', 'tharja pronunciation', 'meaning of the name penelope', 'epitaph meaning']\n",
      "cluster id: 14, queries: ['average price for one person to live in london', 'average starting salary for architects', 'us post stamps price', 'the cost of waiting to invest illustration', 'breast augmentation spain cost', 'cost of average wedding in canada', 'chevrolet avalanche price', 'roofing cost calculator asphalt shingle', 'cost to build a steel building', 'average cost of an engagement ring 2015']\n",
      "cluster id: 15, queries: [\"what's the definition of prudent\", 'what federal taxes do employers pay', 'what does it mean when the state pension rate increases', 'what does it cost to renew your license in NC', 'is interest income on recoveries taxable', 'what payment method does costco take', 'what is law regarding personal financial advisor fees', 'Can you get military retirement and va disability at the same time', 'how do baseball scholarships work', 'what is irs form 147c']\n",
      "cluster id: 16, queries: ['who played shaggy in scooby doo', 'which event of the civil rights movement was considered to be the most significant one', 'who is nana', 'population chehalis wa', 'who is thanos rebellious daughter name', 'largest division of pakistan', 'which actor refused his award for the godfather', 'who was the actress in welcome back kotter', 'who is lori taylor', 'cast of dr blake']\n",
      "cluster id: 17, queries: ['real time stock price', 'meaningful use attestation', 'extra fees to be paid after normal delivery in hospital in dubai', 'refrigeration and air conditioning salary', 'forbes most valuable franchises', 'minimum qualification to apply for afcat', 'temperature in amsterdam today', 'california limit on small claims', 'is krispy kreme open 24 hours', 'calucating salary from wage']\n",
      "cluster id: 18, queries: ['can you give dogs acetaminophen', 'what is prolia injection for', 'can you take mucinex and amoxicillin', 'gout medication list', 'how long does temazepam stay in urine', 'skin cancer cream side effects', 'is minoxidil a prescription?', 'how fast does clonidine lower blood pressure', 'what is radadvantage apc', 'how much is the pay walgreens pharmacy']\n",
      "cluster id: 19, queries: ['what is tenifer finish', 'is it safe to drive jeep with faulty input speed sensor', 'what minimum pressure do all unwrapped metal instruments undergo during flash sterilization?', 'what is the name of the fabric used for sofa rather than leather', 'where is arcade building supply located?', 'what is the code to type registered symbol', 'what is a transmission tariff', 'what kind of mixture is bronze', 'How much mortar should be used between bricks?', 'what is the noise level in machine shop']\n",
      "cluster id: 20, queries: ['in which part is malaria caused', 'where is the beehive in catherby', 'where is kerman located', 'where is bolsover castle', 'where is the white sea', 'where is shaq from', 'where is owensville mo', 'where is the baai', \"what is kroger's clinic called\", 'where is south sandwich island']\n",
      "cluster id: 21, queries: ['count how many of each item in excel', 'how much does it cost to set up a cabinet shop', 'how much does petsmart bathers pay an hour', 'how much is the cat dialect study cost', 'what is the minimum of players in baseball and max players', 'on average how many times does a person breathe per minute?', 'how many days in a lunar calendar', 'how much does people make with a business degree in associate?', 'how much does it cost for a professional nose job', 'how much are the tourism and fishing industries worth annually in australian economy']\n",
      "cluster id: 22, queries: ['dinku define', 'define fop', 'ct injury definition', 'definition of seiche', 'definition of civil', 'define inbound violation in basketball', 'LIRA definition', 'piece define', 'definition hypometria', 'individualistic approach definition']\n",
      "cluster id: 23, queries: [\"what was the length of jurgis' jail sentence?\", 'which albums do other artists sing dylan', 'what tv shows did maggie q star in', 'what bar does austin from jeopardy work at', 'when did wright brothers fly', 'what numbers repeat in mega', 'who stars in film black swan ?', 'what is the google site that matches your face to history', 'is elon musk smart?', 'what did picasso himself say about this painting']\n",
      "cluster id: 24, queries: ['what does the name trumaine mean', 'what test does the doctor do for you kidneys', 'does plasmapheresis cause metabolic acidosis?', 'what is a corm rhs', 'where is the bladder located in the female body when pregnant', 'which layer of the endometrium do the uterine glands develop during the secretory phase of the uterine cycle', 'what is ligase', 'what irregular heart rhythms can you feel', 'what body system does fatty deposits belong to', 'how does contraction move through the heart']\n"
     ]
    }
   ],
   "source": [
    "for key, queries in query_list.items():\n",
    "    # print(f'cluster id: {key}, name: {llm_name_cluster(queries)}')\n",
    "    print(f'cluster id: {key}, queries: {queries[:peek_size]}')\n",
    "    # print('-'*10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0. geography  \n",
      "1. counties  \n",
      "2. science  \n",
      "3. duration  \n",
      "4. definitions  \n",
      "5. history  \n",
      "6. nutrition  \n",
      "7. meanings  \n",
      "8. health  \n",
      "9. contact  \n",
      "10. how-to  \n",
      "11. technology  \n",
      "12. medicine  \n",
      "13. meanings  \n",
      "14. cost  \n",
      "15. finance  \n",
      "16. people  \n",
      "17. current  \n",
      "18. medication  \n",
      "19. materials  \n",
      "20. location  \n",
      "21. quantity  \n",
      "22. definitions  \n",
      "23. entertainment  \n",
      "24. anatomy  \n"
     ]
    }
   ],
   "source": [
    "res = llm_name_all_cluster(query_list)\n",
    "print(res)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "id_name_mapping = {\n",
    "    0: 'geography',\n",
    "    2: 'science',\n",
    "    5: 'history',\n",
    "    8: 'health',\n",
    "    11: 'technology',\n",
    "    15: 'finance',\n",
    "    23: 'entertainment',\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "dpr",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
