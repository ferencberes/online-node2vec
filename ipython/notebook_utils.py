import pandas as pd
import plotly.express as px

class PlayerMetaData():
    def __init__(self, handler):
        self.init(handler)
        
    def init(self, handler):
        account2player = {}
        for player, accounts in handler.player_accounts.items():
            for a in accounts:
                account2player[a] = player
        self.id_to_account = dict(zip(handler.account_to_id.values(), handler.account_to_id.keys()))
        self.handler = handler
        self.account2player = account2player

    def get_player_info(self, node_id, date):
        player_name = self.account2player.get(self.id_to_account.get(node_id))
        match_record = "Not found"
        if player_name != None:
            schedule = self.handler.schedule
            daily_df = schedule[schedule["date"]==date]
            match_record = "%s<br>%s" % (player_name, date)
            if len(daily_df) > 0:
                row = daily_df[(daily_df["playerName active"]==player_name) | (daily_df["playerName opponent"]==player_name)]
                if len(row)>0:
                    row = dict(row[["matchHeader","courtName","orderNumber"]].iloc[0])
                    match_record = "%s<br>%s<br>%s<br>%s<br>Match id: %i" % (player_name, date, row["matchHeader"], row["courtName"], row["orderNumber"])
        return match_record
    
def prepare_animation(handler, player_labels, feature_set, delta_time, model_name, dates, x=1, y=2, frame_duration=1000, transition_duration=500, verbose=False):
    pmd = PlayerMetaData(handler)
    player_nodes = list(pd.concat(player_labels)["id"].unique())
    if verbose:
        print("Number of tennis player accounts:", len(player_nodes))
    snapshot_per_day = int(86400/delta_time)
    num_snapshots = len(feature_set)
    embeddings = []
    for i in range(num_snapshots):
        snap_relevant = list(player_labels[i//snapshot_per_day]["id"])
        snap_emb = feature_set[i][1].rename({0:"id"}, axis=1).copy()
        snap_emb = snap_emb[snap_emb["id"].isin(player_nodes)].copy()
        snap_emb["snapshot"] = i
        snap_emb["label"] = snap_emb["id"].apply(lambda x: 1.0 if x in snap_relevant else 0.0)
        embeddings.append(snap_emb)
        if verbose:
            print(len(snap_emb))
    embedding_df = pd.concat(embeddings).reset_index(drop=True)[["id","label","snapshot",x,y]]
    embedding_df["text"] = embedding_df.apply(lambda x: pmd.get_player_info(x["id"], dates[int(x["snapshot"]//snapshot_per_day)]), axis=1)
    range_x=[embedding_df[x].min(), embedding_df[x].max()]
    range_y=[embedding_df[y].min(), embedding_df[y].max()]
    title = "%s representation over time for tennis player Twitter accounts (snapshots length: %i hours)" % (model_name, delta_time/3600)
    fig = px.scatter(embedding_df, x=x, y=y, animation_frame="snapshot", animation_group="id", color="label", hover_name="text", range_x=range_x, range_y=range_y, color_continuous_scale='Bluered')
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = transition_duration
    _ = fig.layout.update(
        title_text=title,
        title_font_size=18, 
        showlegend=False,
        #coloraxis_showscale=False,
    )
    return fig, embedding_df