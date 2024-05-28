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

# Utility function to add logo to the plots
def add_logo_to_plot(fig, logo_path="Logos/3.png", opacity=0.2, size=(270, 270)):
    logo = Image.open(logo_path)
    logo = logo.resize(size, PIL.Image.LANCZOS)
    enhancer = ImageEnhance.Brightness(logo)
    logo = enhancer.enhance(opacity)
    logo_width, logo_height = logo.size
    center_x = (fig.get_figwidth() + logo_width) * 1.0
    center_y = (fig.get_figheight() + logo_height) * 0.5
    fig.figimage(logo, xo=center_x, yo=center_y, origin='upper')

# Function to generate result string
def generate_result_string(last_n_team_games):
    return ''.join(last_n_team_games['result'].values)

# Function to filter games of a team
def filter_team_games(data, team):
    return data[(data['home'] == team) | (data['away'] == team)]

# Function to get the last n games
def get_last_n_games(team_games, n=5):
    completed_games = team_games.dropna(subset=['home_goal', 'away_goal'])
    return completed_games.tail(n)

# Function to determine the result of a game
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

# Function to display the last n games
def display_last_n_games(last_n_team_games):
    columns_to_display = ['home', 'away', 'home_goal', 'away_goal', 'result']
    return last_n_team_games[columns_to_display]

# Function to plot head-to-head comparison
def head_to_head_plot(data, team1, team2):
    team1_wins = len(data[((data['home'] == team1) & (data['home_goal'] > data['away_goal'])) |
                         ((data['away'] == team1) & (data['away_goal'] > data['home_goal']))])
    team2_wins = len(data[((data['home'] == team2) & (data['home_goal'] > data['away_goal'])) |
                         ((data['away'] == team2) & (data['away_goal'] > data['home_goal']))])
    draws = len(data[(data['home'] == team1) & (data['away'] == team2) & (data['home_goal'] == data['away_goal'])] +
                data[(data['home'] == team2) & (data['away'] == team1) & (data['home_goal'] == data['away_goal'])])

    x_labels = ['Wins', 'Draws']
    team1_values = [team1_wins, draws]
    team2_values = [team2_wins, draws]

    fig, ax = plt.subplots()
    bar_width = 0.35
    bar_positions = list(range(len(x_labels)))

    ax.bar(bar_positions, team1_values, bar_width, label=team1, color='purple')
    ax.bar([pos + bar_width for pos in bar_positions], team2_values, bar_width, label=team2, color='green')

    for i, value in enumerate(team1_values):
        ax.annotate(str(value), xy=(bar_positions[i], value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    for i, value in enumerate(team2_values):
        ax.annotate(str(value), xy=(bar_positions[i] + bar_width, value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    add_logo_to_plot(fig)

    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Matches')
    ax.set_title(f'{team1} vs. {team2} NPFL Comparison from 2002/03 Till Date')
    ax.legend()
    st.pyplot(fig)

# Function to plot total goals scored
def total_goals_plot(data, team1, team2):
    team1_goals = data[((data['home'] == team1) | (data['away'] == team1))]
    team2_goals = data[((data['home'] == team2) | (data['away'] == team2))]

    team1_total_goals = team1_goals['home_goal'].sum() + team1_goals['away_goal'].sum()
    team2_total_goals = team2_goals['home_goal'].sum() + team2_goals['away_goal'].sum()

    x_labels = ['Goals Scored']
    team1_values = [team1_total_goals]
    team2_values = [team2_total_goals]

    fig, ax = plt.subplots()
    bar_width = 0.35
    bar_positions = list(range(len(x_labels)))

    ax.bar(bar_positions, team1_values, bar_width, label=team1, color='purple')
    ax.bar([pos + bar_width for pos in bar_positions], team2_values, bar_width, label=team2, color='green')

    for i, value in enumerate(team1_values):
        ax.annotate(str(value), xy=(bar_positions[i], value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    for i, value in enumerate(team2_values):
        ax.annotate(str(value), xy=(bar_positions[i] + bar_width, value), xytext=(0, 3),
                    textcoords='offset points', ha='center', va='bottom')

    add_logo_to_plot(fig)

    ax.set_xticks([pos + bar_width / 2 for pos in bar_positions])
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('Total Goals')
    ax.set_title(f'Total NPFL Goals Scored from 2002/03 Till Date ({team1} vs {team2})')
    ax.legend()
    st.pyplot(fig)

# Function to calculate head-to-head totals
def calculate_head_to_head_totals(data, team1, team2):
    team1_matches = data[(data['home'] == team1) | (data['away'] == team1)]
    team2_matches = data[(data['home'] == team2) | (data['away'] == team2)]

    team1_total_goals_scored = team1_matches[team1_matches['home'] == team1]['home_goal'].sum() + team1_matches[team1_matches['away'] == team1]['away_goal'].sum()
    team1_total_goals_conceded = team1_matches[team1_matches['home'] == team1]['away_goal'].sum() + team1_matches[team1_matches['away'] == team1]['home_goal'].sum()
    
    team2_total_goals_scored = team2_matches[team2_matches['home'] == team2]['home_goal'].sum() + team2_matches[team2_matches['away'] == team2]['away_goal'].sum()
    team2_total_goals_conceded = team2_matches[team2_matches['home'] == team2]['away_goal'].sum() + team2_matches[team2_matches['away'] == team2]['home_goal'].sum()

    average_goals_scored_team1 = team1_total_goals_scored / len(team1_matches)
    average_goals_conceded_team1 = team1_total_goals_conceded / len(team1_matches)
    average_goals_scored_team2 = team2_total_goals_scored / len(team2_matches)
    average_goals_conceded_team2 = team2_total_goals_conceded / len(team2_matches)

    return (average_goals_scored_team1, average_goals_conceded_team1), (average_goals_scored_team2, average_goals_conceded_team2)

# Function to display goals distribution by season
def display_goals_distribution_by_season(data):
    data['season'] = data.index.str[:4]
    goals_by_season = data.groupby('season')[['home_goal', 'away_goal']].sum()

    fig, ax = plt.subplots()
    goals_by_season.plot(kind='bar', stacked=True, ax=ax)

    add_logo_to_plot(fig)

    ax.set_xlabel('Season')
    ax.set_ylabel('Goals')
    ax.set_title('Goals Distribution by Season')
    st.pyplot(fig)

# Function to display goals distribution by season per team
def display_goals_distribution_by_season_per_team(data, team1, team2):
    data['season'] = data.index.str[:4]

    team1_goals_by_season = data[data['home'] == team1].groupby('season')['home_goal'].sum() + \
                            data[data['away'] == team1].groupby('season')['away_goal'].sum()
    team2_goals_by_season = data[data['home'] == team2].groupby('season')['home_goal'].sum() + \
                            data[data['away'] == team2].groupby('season')['away_goal'].sum()

    fig, ax = plt.subplots()
    team1_goals_by_season.plot(kind='bar', color='blue', alpha=0.7, ax=ax, position=0, width=0.4)
    team2_goals_by_season.plot(kind='bar', color='red', alpha=0.7, ax=ax, position=1, width=0.4)

    add_logo_to_plot(fig)

    ax.set_xlabel('Season')
    ax.set_ylabel('Goals')
    ax.set_title(f'Goals Distribution by Season for {team1} and {team2}')
    st.pyplot(fig)

# Main function for the Streamlit app
def main():
    st.title('NPFL Analysis')
    st.sidebar.title('NPFL Analysis')

    teams = data['home'].unique()

    selected_team = st.sidebar.selectbox('Select a team', teams)
    team_games = filter_team_games(data, selected_team)

    last_n_team_games = get_last_n_games(team_games)
    last_n_team_games['result'] = last_n_team_games.apply(lambda row: determine_result(row, selected_team), axis=1)
    result_string = generate_result_string(last_n_team_games)

    st.header(f'Last 5 Games for {selected_team}')
    st.write(display_last_n_games(last_n_team_games))
    st.subheader(f'Results: {result_string}')

    st.sidebar.title('Head-to-Head Comparison')
    team1 = st.sidebar.selectbox('Select Team 1', teams, key='team1')
    team2 = st.sidebar.selectbox('Select Team 2', teams, key='team2')

    if st.sidebar.button('Compare Teams'):
        head_to_head_plot(data, team1, team2)
        total_goals_plot(data, team1, team2)

    if st.sidebar.button('Show Goals Distribution by Season'):
        display_goals_distribution_by_season(data)

    if st.sidebar.button('Show Goals Distribution by Season for Selected Teams'):
        display_goals_distribution_by_season_per_team(data, team1, team2)

if __name__ == "__main__":
    main()
