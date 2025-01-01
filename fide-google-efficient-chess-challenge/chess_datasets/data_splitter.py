filename = "../raw_data/ficsgamesdb_2023_standard2000_nomovetimes_405168.pgn"
directory = "../data"

def main(max_games_per_file):
    with open(filename, "r") as f:
        lines_counted = 0
        games_counted = 0
        for counter in f:
            if "Event" in counter:
                games_counted += 1
            lines_counted += 1
    print(f"total games found: {games_counted}")
    print(f"total lines found: {lines_counted}")

    with open(filename, "r") as f:
        data_split = []
        games_parsed = 0
        batch_number = 0
        lines_parsed = 0

        for line in f:
            lines_parsed += 1
            if "Event" in line:
                games_parsed += 1
            if games_parsed > max_games_per_file or lines_parsed == lines_counted:
                if lines_parsed == lines_counted:
                    data_split.append(line)
                batch_number += 1
                total = " ".join(data_split)

                with open(f"{directory}/training_batch_{batch_number}.txt", "w+") as newfile:
                    newfile.write(total)
                    print("batch number {} saved".format(batch_number))
                data_split = [line]
                games_parsed = 1
            else:
                data_split.append(line)

    batch_number = 0
    line_checker = 0
    total_game_checker = 0
    while True:
        batch_number += 1
        game_checker = 0
        try:
            with open(f"{directory}/training_batch_{batch_number}.txt", "r") as f:
                for line in f:
                    if "Event" in line:
                        game_checker += 1
                        total_game_checker += 1
                    line_checker += 1
            print("games in file {}: {}".format(batch_number, game_checker))
        except FileNotFoundError:
            if total_game_checker == games_counted:
                print("all games parsed")
            else:
                print("ERROR: not all games parsed")
            if line_checker == lines_counted:
                print("all lines parsed")
            else:
                print("ERROR: not all lines parsed")
            break

if __name__ == "__main__":
    main(300)