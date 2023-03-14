import csv
import logging
import pickle
from typing import Optional

from squat_wars.player import Player


class RankingRow:
    def __init__(self, player: Player, score: int):
        self.player = player
        self.score = score

    def __repr__(self):
        return f"{self.player.name} {self.score}"

    def __lt__(self, other: "RankingRow"):
        return self.score < other.score

    def __gt__(self, other: "RankingRow"):
        return self.score > other.score

    def __eq__(self, other: "RankingRow"):
        return self.score == other.score

    def __str__(self):
        return f"{self.player.name}: {self.player.email} - {self.score}"


class Ranking:
    def __init__(self):
        self.rows: list[RankingRow] = []

    def add(self, player: Player, score: int):
        self.rows.append(RankingRow(player, score))
        self.sort()

    def sort(self):
        self.rows.sort(reverse=True)

    def get_player_position(self, player: Player) -> Optional[int]:
        runs = [(i + 1, row) for i, row in enumerate(self.rows) if row.player == player]
        if runs:
            sort_runs = sorted(runs, key=lambda x: x[1].player.timestamp, reverse=True)
            return sort_runs[0][0]

        return None

    @property
    def highscore(self) -> Optional[int]:
        return self.rows[0].score if self.rows else None

    def __repr__(self):
        return f"{self.rows}"

    def __getitem__(self, key):
        return self.rows[key]

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __str__(self):
        self.sort()
        # Print every ranking row with its number in the ranking
        ret = "\n"
        for i, row in enumerate(self.rows):
            ret += f"{i + 1}. {row}\n"
        return ret

    def __ne__(self, other):
        return self.rows != other.rows

    def save_to_pickle(self):
        with open("ranking.pickle", "wb") as f:
            pickle.dump(self, f)

    def save_to_csv(self):
        with open("ranking.csv", mode="w") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "email", "timestamp", "score"])
            writer.writeheader()
            for x in self.rows:
                writer.writerow(
                    {
                        "name": x.player.name,
                        "email": x.player.email,
                        "timestamp": x.player.timestamp,
                        "score": x.score,
                    }
                )

    @staticmethod
    def load_from_csv(filename: str = "ranking.csv"):
        ranking = Ranking()
        try:
            with open(filename, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player = Player(row["name"], row["email"], int(row["timestamp"]))
                    ranking.add(player, int(row["score"]))
        except FileNotFoundError:
            logging.info("Ranking not found, a new one was created")

        return ranking

    @staticmethod
    def load_from_pickle(filename: str = "ranking.pickle"):
        # check if file doesnt exist
        try:
            with open(filename, "rb") as f:
                ranking = pickle.load(f)
        except FileNotFoundError:
            ranking = Ranking()

        return ranking
