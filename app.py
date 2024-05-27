


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import PIL
from PIL import Image, ImageEnhance
import seaborn as sns

# Step 1: Load data from Google Sheets
document_id = '1rlP9mSfvJE73xK6Q5ddtDnk67U6D1DjVsJH-1dIXOXk'
sheet_name = 'All-Time-Results'  # Replace with the appropriate sheet name or index if necessary
url = f'https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'

data = pd.read_csv(url, index_col=0)

# Step 4: Create the Streamlit app
def main():
    st.title("NPFL ALL TIME ANALYSIS")

    # Step 5: Select two teams for analysis
    unique_home_teams = data['home'].unique()
    unique_away_teams = data['away'].unique()

    # Remove home teams already present in the away teams list
    unique_away_teams = [team for team in unique_away_teams if team not in unique_home_teams]

    # Concatenate unique home and away team names
    all_unique_teams = list(unique_home_teams) + list(unique_away_teams)

    # Use the multiselect widget with the combined unique team names
    selected_teams = st.multiselect("Select Teams", all_unique_teams)

    if len(selected_teams) != 2:
        st.warning("Please select exactly two teams.")
        return

    team1, team2 = selected_teams

    # Step 6: Filter the data based on the selected teams
    team_data = data[(data['home'].isin(selected_teams)) & (data['away'].isin(selected_teams))]

    # Step 7: Display the raw data for the selected teams
    if st.checkbox("Show Raw Data"):
        st.dataframe(team_data[['home', 'away', 'home_goal', 'away_goal']])

    # Step 8: Display the head-to-head comparison
    st.subheader("Head-to-Head Comparison form 2003 Till Date")
    head_to_head_plot(team_data, team1, team2)

    # Step 9: Display total goals scored by each team
    st.subheader("Total Goals Scored from 2023 Till Date")
    total_goals_plot(data, team1, team2)

    st.header("Average Goals Analysis")
    for team in selected_teams:
        st.write(f"**{team}**:")
        st.write(f"Average Goals Scored: {calculate_average_goals(team_data, team, is_home_team=True):.2f}")
        st.write(f"Average Goals Conceded: {calculate_average_goals(team_data, team, is_home_team=False):.2f}")

    st.title("Goals Distribution by Season")
    # Group data by team and season, and sum the goals
    goals_distribution = team_data.groupby(['home', 'season'])[['home_goal', 'away_goal']].sum().reset_index()

    # Sum the total goals (home_goal + away_goal)
    goals_distribution['total_goals'] = goals_distribution['home_goal'].astype(int) + goals_distribution['away_goal'].astype(int)
    goals_distribution['leg'] = goals_distribution.groupby('season').cumcount() + 1
    goals_distribution['leg'] = goals_distribution['leg'].replace({1: 'First Leg', 2: 'Second Leg'})

    # Print out the number of goals per season
    st.title("Goals Distribution by Season")
    # Group data by team and season, and sum the goals
    goals_distribution = team_data.groupby(['home', 'season'])[['home_goal', 'away_goal']].sum().reset_index()

    # Sum the total goals (home_goal + away_goal)
    goals_distribution['total_goals'] = goals_distribution['home_goal'].astype(int) + goals_distribution['away_goal'].astype(int)
    goals_distribution['leg'] = goals_distribution.groupby('season').cumcount() + 1
    goals_distribution['leg'] = goals_distribution['leg'].replace({1: 'First Leg', 2: 'Second Leg'})

    # Print out the number of goals per season
    st.write("Number of goals per season:")
    st.table(goals_distribution[['season', 'total_goals']].groupby('season').sum().astype(int))

    # Filter to include only the last 10 seasons
    last_10_seasons = goals_distribution['season'].unique()[-10:]
    goals_distribution_last_10 = goals_distribution[goals_distribution['season'].isin(last_10_seasons)]

    # Sort the data by season to ensure chronological order
    goals_distribution_last_10 = goals_distribution_last_10.sort_values(by='season')

    # Plot the bar chart for the last 10 seasons
    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='total_goals', hue="leg", data=goals_distribution_last_10, ci=None)
    plt.title("Goals Distribution by Season (Last 10 Seasons)")
    plt.xlabel("Season")
    plt.ylabel("Total Goals")

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)  # ha='right' for better alignment, fontsize can be adjusted if needed

    st.pyplot(plt)

    st.title("Goals Distribution by Season per Team")
    goals_distribution_per_team = team_data.groupby(['season', 'home'])[['home_goal']].sum().reset_index()
    goals_distribution_per_team = goals_distribution_per_team.rename(columns={'home': 'team', 'home_goal': 'goals_scored'})
    
    # Sum away goals per team and merge with home goals
    away_goals_per_team = team_data.groupby(['season', 'away'])[['away_goal']].sum().reset_index()
    away_goals_per_team = away_goals_per_team.rename(columns={'away': 'team', 'away_goal': 'goals_scored'})
    
    goals_distribution_per_team = pd.concat([goals_distribution_per_team, away_goals_per_team], axis=0)
    goals_distribution_per_team = goals_distribution_per_team.groupby(['season', 'team'])[['goals_scored']].sum().reset_index()

    st.write("Number of goals per season per team:")
    st.table(goals_distribution_per_team.pivot(index='season', columns='team', values='goals_scored').fillna(0).astype(int))

    goals_distribution_per_team_last_10 = goals_distribution_per_team[goals_distribution_per_team['season'].isin(last_10_seasons)]

    plt.figure(figsize=(10, 6))
    sns.barplot(x='season', y='goals_scored', hue='team', data=goals_distribution_per_team_last_10, ci=None)
    plt.title("Goals Distribution by Season per Team (Last 10 Seasons)")
    plt.xlabel("Season")
    plt.ylabel("Total Goals")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    st.pyplot(plt)

    team1_games = filter_team_games(data, team1)
    team2_games = filter_team_games(data, team2)

    last_5_team1_games = get_last_n_games(team1_games)
    last_5_team2_games = get_last_n_games(team2_games)

    last_5_team1_games['result'] = last_5_team1_games.apply(determine_result, axis=1, team=team1)
    last_5_team2_games['result'] = last_5_team2_games.apply(determine_result, axis=1, team=team2)

    st.subheader(f"Last 5 games involving {team1}")
    st.write(display_last_n_games(last_5_team1_games))

    st.subheader(f"Last 5 games involving {team2}")
    st.write(display_last_n_games(last_5_team2_games))

# Step 10: Data Visualization functions
def filter_team_games(data, team):
    return data[(data['home'] == team) | (data['away'] == team)]

def get_last_n_games(team_games, n=5):
    # Filter out games that haven't been played yet (missing goals)
    completed_games = team_games.dropna(subset=['home_goal', 'away_goal'])
    return completed_games.tail(n)

def determine_result(row, team):
    if team == row['home']:
        if row['home_goal'] > row['away_goal']:
            return 'W'
        elif row['home_goal'] < row['away_goal']:
            return 'L'
        else:
            return 'D'
    elif team == row['away']:
        if row['away_goal'] > row['home_goal']:
            return 'W'
        elif row['away_goal'] < row['home_goal']:
            return 'L'
        else:
            return 'D'
    else:
        return None

def display_last_n_games(last_n_team_games):
    columns_to_display = ['home', 'away', 'home_goal', 'away_goal', 'result']
    return last_n_team_games[columns_to_display]

def head_to_head_plot(data, team1, team2):
    home_wins_team1 = data[(data['home'] == team1) & (data['home_goal'] > data['away_goal'])]
    away_wins_team1 = data[(data['away'] == team1) & (data['away_goal'] > data['home_goal'])]
    draws_team1 = data[((data['home'] == team1) | (data['away'] == team1)) & (data['home_goal'] == data['away_goal'])]

    home_wins_team2 = data[(data['home'] == team2) & (data['home_goal'] > data['away_goal'])]
    away_wins_team2 = data[(data['away'] == team2) & (data['away_goal'] > data['home_goal'])]
    draws_team2 = data[((data['home'] == team2) | (data['away'] == team2)) & (data['home_goal'] == data['away_goal'])]

    x_labels = ['Wins', 'Draws']
    team1_values = [len(home_wins_team1) + len(away_wins_team1), len(draws_team1)]
    team2_values = [len(home_wins_team2) + len(away_wins_team2), len(draws_team2)]

    fig, ax = plt.subplots()
    bar_width = 0.15
    bar_positions = list(range(len(x_labels)))

    ax.bar(bar_positions, team1_values, bar_width, label=team1, color=['purple'])
    ax.bar([pos + bar_width for pos in bar_positions], team2_values, bar_width, label=team2)

    for i, value in enumerate(team1_values):
        ax.annotate(str(value), xy=(bar_positions[i], value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    for i, value in enumerate(team2_values):
        ax.annotate(str(value), xy=(bar_positions[i] + bar_width, value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    team1_matches = data[(data['home'] == team1) | (data['away'] == team1)]
    team2_matches = data[(data['home'] == team2) | (data['away'] == team2)]
    total_matches = (int(len(team1_matches)) + int(len(team2_matches))) / 2

    logo = Image.open("Logo.png")
    logo = logo.resize((400, 400), PIL.Image.Resampling.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo)
    logo = enhancer.enhance(opacity)

    logo_width, logo_height = logo.size
    center_x = (fig.get_figwidth() + logo_width) * 1.0
    center_y = (fig.get_figheight() + logo_height) * 0.5

    # Place the logo in the middle
    ax.figure.figimage(logo, xo=center_x, yo=center_y, origin='upper')

    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Matches')
    ax.set_title(f'{team1} vs. {team2} Head-to-Head Comparison from 2023 Till Date')
    ax.legend()
    st.write(f"Total matches played by {team1} and {team2}: {total_matches}")
    st.pyplot(fig)

def total_goals_plot(data, team1, team2):
    team1_goals = data[((data['home'] == team1) & (data['away'] == team2)) | ((data['home'] == team2) & (data['away'] == team1))]
    team2_goals = data[((data['home'] == team2) & (data['away'] == team1)) | ((data['home'] == team1) & (data['away'] == team2))]

    team1_home_data = team1_goals[team1_goals['home'] == team1]
    team1_away_data = team1_goals[team1_goals['away'] == team1]
    team1_score = team1_home_data['home_goal'].sum() + team1_away_data['away_goal'].sum()

    team2_home_data = team2_goals[team2_goals['home'] == team2]
    team2_away_data = team2_goals[team2_goals['away'] == team2]
    team2_score = team2_home_data['home_goal'].sum() + team2_away_data['away_goal'].sum()

    st.write(team1 + " total goals against " + team2, team1_score)
    st.write(team2 + " total goals against " + team1, team2_score)

    x_labels = [team1, team2]
    y_values = [team1_score, team2_score]

    bar_width = 0.15
    bar_positions = list(range(len(x_labels)))

    fig, ax = plt.subplots(figsize=(4, 3))

    logo = Image.open("Logo.png")
    logo = logo.resize((270, 270), PIL.Image.LANCZOS)
    opacity = 0.2
    enhancer = ImageEnhance.Brightness(logo)
    logo = enhancer.enhance(opacity)

    logo_width, logo_height = logo.size
    center_x = (fig.get_figwidth() + logo_width) * 1.0
    center_y = (fig.get_figheight() + logo_height) * 0.5

    # Place the logo in the middle
    ax.figure.figimage(logo, xo=center_x, yo=center_y, origin='upper')
    ax.bar(x_labels, y_values, width=bar_width, color=['darkblue', 'purple'])
    for i, value in enumerate(y_values):
        # ax.text(i, value + 1, value, ha='center')
        pass

    ax.set_ylabel('Total Goals Scored')
    st.pyplot(fig)

def calculate_average_goals(team_data, team_name, is_home_team=True):
    if is_home_team:
        goals_column = 'home_goal'
    else:
        goals_column = 'away_goal'

    total_goals = team_data[team_data['home' if is_home_team else 'away'] == team_name][goals_column].sum()
    total_matches = len(team_data[team_data['home' if is_home_team else 'away'] == team_name])

    if total_matches == 0:
        return 0

    average_goals = total_goals / total_matches
    return average_goals

# Step 11: Run the app
if __name__ == "__main__":
    main()
