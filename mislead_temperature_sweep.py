from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from storyboard import Storyboard
from storysim import StorySimulator
import pandas as pd
from experiment_defs import mislead_experiment

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

client = OpenAI()


def prompt_model(prompt, model, temperature=0.0):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def compute_score_unsure(label, response, starting_loc='hallway'):
    base = f'{label.split("_")[0]}_' if not label.startswith(starting_loc) else label
    response = response.split("\n")[-1]
    response = response.lower().replace("_",' ')
    if response.count(base) <= 1:
        return str(label in response or label.replace('_'," ") in response or 
                   label.replace('_',"") in response or
                   (starting_loc in label and starting_loc in response) or
                   label.replace(" ","") in response)
    elif 'Therefore,' in response:
        response = response.split('Therefore,')[-1]
        return str(label in response or label.replace('_'," ") in response or label.replace('_',"") in response or (starting_loc in label and starting_loc in response))
    return f'{response}, {label}'


if __name__ == '__main__':
    # Settings
    models = ['gpt-4']
    temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
    values = [5, 30, 80]  # mislead distances to sweep
    num_trials = 50
    num_people = 7

    out_dir = os.path.expanduser('~/scratch/heuristics-temp-sweep')
    os.makedirs(out_dir, exist_ok=True)

    random.seed(25)

    for temp in temperatures:
        print(f"\nRunning temperature={temp}")
        all_rows = []

        for v in values:
            print(f"  Mislead distance = {v}")
            df = pd.DataFrame({'Story':[], 'Label':[], 'P1':[], 'P2':[]})
            graph = {
                "room_1": ["room_2", "the_hallway","room_5"],
                "room_2": ["room_1", "room_3","the_hallway"],
                "room_3": ["room_2", "room_4","the_hallway"],
                "room_4": ["room_3", "room_5","room_1"],
                "room_5": ["room_4", "room_1","room_2"],
                "the_hallway": ["room_1", "room_4","room_2"]
            }

            locations = list(graph.keys())
            story_length = 100

            for _ in range(num_trials):
                event_dict, max_actor = mislead_experiment(v, story_length)
                storyboard = Storyboard('enters', graph, [f'P{i}' for i in range(num_people)], story_length, event_dict)

                sim = StorySimulator(
                    people=[f'P{i}' for i in range(num_people)],
                    locations=locations,
                    action='enters',
                    storyboard=storyboard,
                    graph=graph
                )

                res = sim.run_simulation(story_length)
                story = sim.formal_to_story(res)

                d = {'Story':[], 'Label':[], 'P1':[], 'P2':[]}
                d['P1'] = ','.join([storyboard.actor_mapping[str(a)] for a in range(int(max_actor))])
                d['P2'] = storyboard.actor_mapping[max_actor]
                d['Story'].append(story)
                d['Label'].append(storyboard.loc_mapping[sorted(storyboard.loc_mapping.keys())[-2]])
                df = pd.concat([df, pd.DataFrame(d)])

            # Query model for each row
            responses = []
            for i, (_, row) in enumerate(df.iterrows()):
                if i % 10 == 0:
                    print(f'    Querying row {i}/{len(df)}')
                p = row['P1'].split(',')
                prompt_prefix = f"Read the following story and answer the question at the end. Note that all characters start in {sim.locations[-1].replace('_',' ')}. Characters in the same location can see where eachother go when someone leaves. If characters are in different locations, they cannot see eachother. There is enough information to answer every question.\n{row['Story']}.\n"
                prompt = f"{prompt_prefix}Q: Where does {p[0]} think {row['P2']} is?\nA:"
                resp = prompt_model(prompt, models[0], temperature=temp)
                responses.append(resp)

            df['Response'] = responses
            df['Score'] = df.apply(lambda r: compute_score_unsure(r['Label'], r['Response']), axis=1)
            correct = sum(df['Score'] == 'True')
            total = len(df)
            acc = correct / total * 100 if total else 0
            print(f'    Accuracy @ temp={temp}, mislead={v}: {correct}/{total} ({acc:.1f}%)')

            df['Mislead'] = v
            df['Temperature'] = temp
            all_rows.append(df)

        # concat all mislead values for this temperature and save
        out_df = pd.concat(all_rows, ignore_index=True)
        out_path = os.path.join(out_dir, f'gpt4_temp_{str(temp).replace(".","_")}.csv')
        out_df.to_csv(out_path, index=False)
        print(f'  Saved results to {out_path}')
