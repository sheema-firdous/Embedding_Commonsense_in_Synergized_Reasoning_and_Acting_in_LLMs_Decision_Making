#!/usr/bin/env python
# coding: utf-8

# In[1]:


import alfworld
print("ALFWorld imported successfully!")


# In[1]:


import os
import json
import glob
import random
from os.path import join as pjoin

import textworld
from textworld.agents import HumanAgent
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType

class Args:
    problem = None  # leave None to pick a random problem
    domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
    grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
    expert = "heuristic"  # or "planner"

args = Args()

# Pick a random problem if None
if args.problem is None:
    problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
    problems = [p for p in problems if "movable_recep" not in p]
    if len(problems) == 0:
        raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
    args.problem = os.path.dirname(random.choice(problems))

if "movable_recep" in args.problem:
    raise ValueError("This problem contains movable receptacles, which is not supported by ALFWorld.")


# In[3]:


print(f"Playing '{args.problem}'.")
GAME_LOGIC = {
    "pddl_domain": open(args.domain).read(),
    "grammar": open(args.grammar).read(),
}

# load state and trajectory files
pddl_file = os.path.join(args.problem, 'initial_state.pddl')
json_file = os.path.join(args.problem, 'traj_data.json')
with open(json_file, 'r') as f:
    traj_data = json.load(f)
GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)

# dump game file
gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
json.dump(gamedata, open(gamefile, "w"))

# expert agent
if args.expert == "planner":
    expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
elif args.expert == "heuristic":
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
else:
    raise ValueError(f"Unknown expert type: {args.expert}")

# register Gym environment
request_infos = textworld.EnvInfos(
    won=True, 
    admissible_commands=True, 
    score=True, 
    max_score=True, 
    intermediate_reward=True, 
    extras=["expert_plan"]
)
env_id = textworld.gym.register_game(
    gamefile, 
    request_infos,
    max_episode_steps=1000000,
    wrappers=[AlfredDemangler(), expert]
)
env = textworld.gym.make(env_id)
obs, infos = env.reset()


# In[4]:


agent = HumanAgent(autocompletion=True, oracle=(expert.expert_type==AlfredExpertType.PLANNER))
agent.reset(env)

done = False
while not done:
    print("\nObservation:\n", obs)
    cmd = input("Your command: ")
    
    if cmd.strip() == "":
        print("Please enter a command.")
        continue
    if cmd == "ipdb":
        from ipdb import set_trace; set_trace()
        continue
    
    obs, score, done, infos = env.step(cmd)
    if done:
        print("🎉 You won!")
        break


# In[ ]:





# In[12]:


from groq import Groq

client = Groq(api_key="API")

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Your task is to look for the newspaper, your options are to goto fridge or to coffeetable. How would you do?"}
    ],
)

print(completion.choices[0].message.content)


# In[ ]:





# # ALFWorld: Calling LLM in the Loop 

# In[7]:



import os
import json
import glob
import random
from os.path import join as pjoin

import textworld
from textworld.agents import HumanAgent
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType
from groq import Groq

# Initialize Groq client
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

class Args:
    problem = None  # leave None to pick a random problem
    domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
    grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
    expert = "heuristic"  # or "planner"

args = Args()

# Pick a random problem if None
if args.problem is None:
    problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
    problems = [p for p in problems if "movable_recep" not in p]
    if len(problems) == 0:
        raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")
    args.problem = os.path.dirname(random.choice(problems))

if "movable_recep" in args.problem:
    raise ValueError("This problem contains movable receptacles, which is not supported by ALFWorld.")

print(f"Playing '{args.problem}'.")
GAME_LOGIC = {
    "pddl_domain": open(args.domain).read(),
    "grammar": open(args.grammar).read(),
}

# load state and trajectory files
pddl_file = os.path.join(args.problem, 'initial_state.pddl')
json_file = os.path.join(args.problem, 'traj_data.json')
with open(json_file, 'r') as f:
    traj_data = json.load(f)
GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)

# dump game file
gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
json.dump(gamedata, open(gamefile, "w"))

# expert agent
if args.expert == "planner":
    expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
elif args.expert == "heuristic":
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
else:
    raise ValueError(f"Unknown expert type: {args.expert}")

# register Gym environment
request_infos = textworld.EnvInfos(
    won=True, 
    admissible_commands=True, 
    score=True, 
    max_score=True, 
    intermediate_reward=True, 
    extras=["expert_plan"]
)
env_id = textworld.gym.register_game(
    gamefile, 
    request_infos,
    max_episode_steps=1000000,
    wrappers=[AlfredDemangler(), expert]
)
env = textworld.gym.make(env_id)
obs, infos = env.reset()

agent = HumanAgent(autocompletion=True, oracle=(expert.expert_type==AlfredExpertType.PLANNER))
agent.reset(env)

# Track game state for context
game_history = []
current_task = ""

# Function to get LLM's observation
def get_llm_observation(observation, task):
    llm_prompt = f"""
Current task: {task}
Game state: {observation}

Based on the current state of the game and the task to complete, provide:
1. A clear observation of what the player can see
2. What might be important about this location
3. What the player should consider doing next

Format your response as a concise observation paragraph.
"""
    
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for the ALFWorld game. Provide clear, concise observations about the game state."},
            {"role": "user", "content": llm_prompt}
        ],
    )
    
    return completion.choices[0].message.content

done = False
while not done:
    # Extract the task if not already captured
    if not current_task and "Your task is to:" in obs:
        task_start = obs.find("Your task is to:") 
        task_end = obs.find("\n", task_start)
        if task_end == -1:  # If no newline after task
            task_end = len(obs)
        current_task = obs[task_start:task_end].strip()
    
    # Update game history
    game_history.append(f"Game: {obs}")
    
    # Print the original, raw observation from the game first
    print("\n" + "="*40)
    print("Raw Game Observation:\n", obs)
    print("="*40)
    
    # Display the current task/goal prominently
    task_display = current_task.replace("Your task is to:", "").strip()
    print("\n🎯 GOAL: " + task_display)
    print("-"*40)
    
    # Get and display LLM's observation
    llm_observation = get_llm_observation(obs, current_task)
    print("\n🤖 LLM Narrator:\n", llm_observation)
    
    # Get player command
    cmd = input("\nYour command: ")
    
    if cmd.strip() == "":
        print("Please enter a command.")
        continue
    if cmd == "ipdb":
        from ipdb import set_trace; set_trace()
        continue
    if cmd == "raw":
        # Option to see the raw observation from the game
        print("\nRaw Game Observation:\n", obs)
        continue
    
    # Update game history with player's command
    game_history.append(f"Player: {cmd}")
    
    # Process the command in the game
    obs, score, done, infos = env.step(cmd)
    
    if done:
        # Final observation with congratulations from LLM
        game_history.append(f"Game: {obs}")
        
        # Print the original, raw final observation
        print("\n" + "="*40)
        print("Raw Final Observation:\n", obs)
        print("="*40)
        
        # Display the completed goal
        print("\n🎯 COMPLETED GOAL: " + task_display)
        print("-"*40)
        
        final_prompt = f"""
Current task: {current_task}
Game state: {obs}

The player has completed the game. Provide a final observation and congratulate them.
"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the ALFWorld game."},
                {"role": "user", "content": final_prompt}
            ],
        )
        
        print("\n🏆 LLM Finale:\n", completion.choices[0].message.content)
        print("\n🎉 You won!")
        break


# 

# In[6]:


get_ipython().system('pip show alfworld')


# # Calling CONCEPTNET in REAL

# In[33]:


import os
import json
import glob
import random
import re
import time
from os.path import join as pjoin

import textworld
from textworld.agents import HumanAgent
from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType

from groq import Groq
import requests
from bs4 import BeautifulSoup
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================
#                 LLaMA + ConceptNet Agent
# =====================================================

# -------------------------------
# LLaMA API Setup
# -------------------------------
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# =====================================================
# CONCEPTNET SCRAPER SECTION (embedded)
# =====================================================
ALL_RELATIONS = [
    "IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation",
    "HasSubevent", "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite",
    "HasProperty", "MotivatedByGoal", "CreatedBy", "LocatedNear",
    "MadeOf", "SimilarTo"
]
EDGE_LIMITS = [5]  # edges per relation
NUM_WORKERS = 8

def parse_relations_from_html(html):
    """Parse relations and edges from ConceptNet HTML."""
    soup = BeautifulSoup(html, "html.parser")
    relations_dict = {rel: [] for rel in ALL_RELATIONS}

    for box in soup.find_all("div", class_="feature-box"):
        h2 = box.find("h2")
        if h2 and h2.a:
            rel_link = h2.a.get("href", "")
            if "rel=" in rel_link:
                rel_name = rel_link.split("rel=")[1].split("&")[0].replace("/r/", "")
                if rel_name in ALL_RELATIONS:
                    edges = [
                        li.find("a").text.strip()
                        for li in box.find_all("li", class_="term") if li.find("a")
                    ]
                    if edges:
                        relations_dict[rel_name] = edges
    return relations_dict

def fetch_relation_edges_parallel(rel_name, edges, edge_limit):
    """Return top `edge_limit` edges and measure latency per relation."""
    start = time.perf_counter()
    limited_edges = edges[:edge_limit] if edges else []
    latency = time.perf_counter() - start
    return {"relation": rel_name, "edges": limited_edges, "latency": latency}

def fetch_concept_edges_parallel(concept):
    """Fetch ConceptNet relations for a given concept and print results."""
    url = f"https://conceptnet.io/c/en/{concept}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    start_total = time.perf_counter()
    response = requests.get(url, headers=headers)
    page_latency = time.perf_counter() - start_total

    all_relations = parse_relations_from_html(response.text)

    results_table = {}
    for edge_limit in EDGE_LIMITS:
        results = []
        start_edges = time.perf_counter()

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [
                ex.submit(fetch_relation_edges_parallel, rel, edges, edge_limit)
                for rel, edges in all_relations.items()
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

        table = PrettyTable(["Relation", f"Edges (limit={edge_limit})", "Edges Count"])
        for r in sorted(results, key=lambda x: x["relation"]):
            table.add_row([
                r["relation"],
                ", ".join(r["edges"]) if r["edges"] else "-",
                len(r["edges"])
            ])

        results_table[edge_limit] = table

#     print(f"\n⏱ ConceptNet fetch latency for '{concept}': {page_latency:.4f} seconds\n")
    for limit, table in results_table.items():
        print(f"=== EDGE LIMIT: {limit} ===")
        print(table)
        print()

# =====================================================
# HELPER: Extract key objects from LLaMA reasoning
# =====================================================
def extract_objects_from_text(text):
    """Extract potential nouns (simple heuristic)."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {"the", "and", "you", "can", "see", "with", "this", "that",
                 "there", "what", "should", "for", "are", "have", "from", "into"}
    objects = [w for w in words if w not in stopwords]
    return list(dict.fromkeys(objects))[:3]  # unique, top 3

# =====================================================
# SETUP ALFWORLD GAME
# =====================================================
class Args:
    problem = None
    domain = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
    grammar = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
    expert = "heuristic"

args = Args()

if args.problem is None:
    problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)
    problems = [p for p in problems if "movable_recep" not in p]
    if not problems:
        raise ValueError("No ALFWorld problem files found. Run `alfworld-data` first.")
    args.problem = os.path.dirname(random.choice(problems))

if "movable_recep" in args.problem:
    raise ValueError("This problem contains movable receptacles, unsupported by ALFWorld.")

print(f"🎮 Starting ALFWorld problem: '{args.problem}'")

GAME_LOGIC = {
    "pddl_domain": open(args.domain).read(),
    "grammar": open(args.grammar).read(),
}

pddl_file = os.path.join(args.problem, 'initial_state.pddl')
json_file = os.path.join(args.problem, 'traj_data.json')
with open(json_file, 'r') as f:
    traj_data = json.load(f)
GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)

gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
json.dump(gamedata, open(gamefile, "w"))

if args.expert == "planner":
    expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
else:
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

request_infos = textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    score=True,
    max_score=True,
    intermediate_reward=True,
    extras=["expert_plan"]
)
env_id = textworld.gym.register_game(
    gamefile,
    request_infos,
    max_episode_steps=1000000,
    wrappers=[AlfredDemangler(), expert]
)
env = textworld.gym.make(env_id)
obs, infos = env.reset()

agent = HumanAgent(autocompletion=True, oracle=(expert.expert_type==AlfredExpertType.PLANNER))
agent.reset(env)

# =====================================================
# MAIN GAME LOOP
# =====================================================
game_history = []
current_task = ""
done = False

import re
import requests
from bs4 import BeautifulSoup
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# =====================================
# CONCEPTNET SCRAPER (copied directly from your reference)
# =====================================
ALL_RELATIONS = [
    "IsA", "PartOf", "HasA", "UsedFor", "CapableOf", "AtLocation",
    "HasSubevent", "HasFirstSubevent", "HasLastSubevent", "HasPrerequisite",
    "HasProperty", "MotivatedByGoal", "CreatedBy", "LocatedNear",
    "MadeOf", "SimilarTo"
]
EDGE_LIMITS = [7]
NUM_WORKERS = 8

def parse_relations_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    relations_dict = {rel: [] for rel in ALL_RELATIONS}
    for box in soup.find_all("div", class_="feature-box"):
        h2 = box.find("h2")
        if h2 and h2.a:
            rel_link = h2.a.get("href", "")
            if "rel=" in rel_link:
                rel_name = rel_link.split("rel=")[1].split("&")[0].replace("/r/", "")
                if rel_name in ALL_RELATIONS:
                    edges = [li.find("a").text.strip() for li in box.find_all("li", class_="term") if li.find("a")]
                    if edges:
                        relations_dict[rel_name] = edges
    return relations_dict

def fetch_relation_edges_parallel(rel_name, edges, edge_limit):
    start = time.perf_counter()
    limited_edges = edges[:edge_limit] if edges else []
    latency = time.perf_counter() - start
    return {"relation": rel_name, "edges": limited_edges, "latency": latency}

def fetch_concept_edges_parallel(concept):
    url = f"https://conceptnet.io/c/en/{concept}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    start_total = time.perf_counter()
    response = requests.get(url, headers=headers)
    page_latency = time.perf_counter() - start_total

    if response.status_code != 200:
        print(f"❌ Failed to fetch ConceptNet for '{concept}' (status={response.status_code})")
        return

    all_relations = parse_relations_from_html(response.text)
    results_table = {}
    for edge_limit in EDGE_LIMITS:
        results = []
        start_edges = time.perf_counter()
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = [ex.submit(fetch_relation_edges_parallel, rel, edges, edge_limit)
                       for rel, edges in all_relations.items()]
            for fut in as_completed(futures):
                results.append(fut.result())

        total_edges = sum(len(r["edges"]) for r in results)
        table = PrettyTable(["Relation", f"Edges (limit={edge_limit})", "Edges Count"])
        for r in sorted(results, key=lambda x: x["relation"]):
            table.add_row([r["relation"], ", ".join(r["edges"]) if r["edges"] else "-", len(r["edges"])])
        results_table[edge_limit] = {"table": table, "total_edges": total_edges}

    print(f"\n🌐 Searching ConceptNet for: {concept}")
    print(f"⏱ Fetch latency: {page_latency:.4f} seconds\n")
    for limit, data in results_table.items():
        print(f"=== EDGE LIMIT: {limit} ===")
        print(data["table"])
        print("\n")


# =====================================
# LLM OBSERVATION + CONCEPTNET REASONING
# =====================================
def get_llm_subgoals(observation, task):
    llm_prompt = f"""
You are an intelligent agent for a TextWorld game. Your job is to decompose a goal into a sequence of actionable sub-goals.

RULES:
1. Identify the main action verbs and modifiers (like 'clean', 'cool', 'put', 'slice') in the task.
2. Identify the main subject (object acted upon) and target object/location.
3. Decompose the task into a **list of ordered sub-goals** reflecting real-world steps to complete the task.
4. Use simple action phrases like: "find X", "clean X", "put X in Y".
5. Output ONLY a Python list variable called `sub_goals`.

Examples:
- Task: "put a clean soapbar in shelf"  
  Output: sub_goals = ["find soapbar", "clean soapbar", "put soapbar in shelf"]

- Task: "put some cool lettuce in garbagecan"  
  Output: sub_goals = ["find lettuce", "cool lettuce", "put lettuce in garbagecan"]

- Task: "put toiletpaper on toiletpaperhanger"  
  Output: sub_goals = ["find toiletpaper", "put toiletpaper on toiletpaperhanger"]

Task to decompose:
{task}

SCENE (for reference):
{observation}

Output ONLY the Python list as shown in examples.
"""
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a precise task decomposition agent."},
            {"role": "user", "content": llm_prompt}
        ],
    )
    return completion.choices[0].message.content

while not done:
    if not current_task and "Your task is to:" in obs:
        start = obs.find("Your task is to:")
        end = obs.find("\n", start)
        if end == -1:
            end = len(obs)
        current_task = obs[start:end].strip()

    print("\n" + "="*40)
    print("Raw Game Observation:\n", obs)
    print("="*40)

    task_display = current_task.replace("Your task is to:", "").strip()
    print("\n🎯 GOAL: " + task_display)
    print("-"*40)

    # --- LLM extracts objects, action, target directly from the full task ---
    llm_prompt = f"""
You are an assistant for a TextWorld game.
Given a GOAL string, extract:

1. goal_subjects = the main physical objects/items being acted upon.
2. goal_objects = the target location or object affected by the action.
3. goal_actions = verbs/actions being performed (main verb + any modifiers).

Do NOT split into sub-goals. Do not add reasoning or explanations.  
Output valid Python lists.

GOAL: "{task_display}"
"""
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a precise goal parser."},
            {"role": "user", "content": llm_prompt}
        ],
    )

    llm_result = completion.choices[0].message.content
    print("\n🤖 LLM Task Analysis:\n", llm_result)

    # --- Parse lists from LLM output ---
    try:
        goal_subjects = eval(re.search(r"goal_subjects\s*=\s*(\[.*?\])", llm_result).group(1))
        goal_objects = eval(re.search(r"goal_objects\s*=\s*(\[.*?\])", llm_result).group(1))
        goal_actions = eval(re.search(r"goal_actions\s*=\s*(\[.*?\])", llm_result).group(1))
    except:
        goal_subjects, goal_objects, goal_actions = [], [], []

    print(f"\nCurrent objects in scene:\n{current_objects}")
    print(f"Goal subjects: {goal_subjects}")
    print(f"Goal objects/targets: {goal_objects}")
    print(f"Goal actions: {goal_actions}")

    # --- ConceptNet query for the first subject ---
    if goal_subjects:
        concept_to_query = goal_subjects[0]
        print(f"\n🌐 ConceptNet Lookup for: {concept_to_query}")
        fetch_concept_edges_parallel(concept_to_query)

        # --- Suggest location based on intersection with current objects ---
        likely_locations = []
        url = f"https://conceptnet.io/c/en/{concept_to_query}"
        response = requests.get(url)
        if response.status_code == 200:
            all_relations = parse_relations_from_html(response.text)
            atloc_edges = all_relations.get("AtLocation", [])
            likely_locations = [obj for obj in current_objects if any(loc.lower() in obj.lower() for loc in atloc_edges)]

        if likely_locations:
            print(f"💡 Suggested first location to check: {likely_locations[0]}")
        else:
            print("💡 No matching location found in scene; explore manually.")

    # --- Ask for user command ---
    cmd = input("\nYour command: ").strip()
    if not cmd:
        print("Please enter a command.")
        continue

    obs, score, done, infos = env.step(cmd)

    if done:
        print("\n🏆 GAME COMPLETE!")
        break


# In[ ]:




