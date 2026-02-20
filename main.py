import os
import noisy_speech_eval as nse

if __name__ == "__main__":

    # Example implementation of evaluation

    # Reference session files are in the ref_dir directory
    # ASR session files are in the pred_dir directory
    # Names of the files to be compared should be the same for both reference and predicted files

    REF_DIR = "data/reference/"
    PRED_DIR = "data/predicted/"

    # Get the session ids from the reference folder
    session_ids = [item.split(".")[0] for item in os.listdir(REF_DIR)]
    benchmarks = []

    for session in session_ids:
        print(f"Processing session {session}")

        # # Create session objects for both reference and predicted files
        truth = nse.Session.from_chat(f"{REF_DIR}{session}.cha")
        pred = nse.Session.from_chat(f"{PRED_DIR}{session}.cha")

        # Create a benchmark object to compare the two sessions. Append the benchmark object to the list of benchmarks
        benchmark = nse.Benchmark(reference=truth, prediction=pred, corpus="APROCSA")
        # print(benchmark.calculate_asr_performance())
        benchmarks.append(benchmark)

    # Combine the results of all the ASR benchmarks and plot the results
    df, plt, df_meta = nse.combined_asr_performance(benchmarks)
    plt.savefig("asr_out_ALL.png")
    plt.clf()

    print(df.to_string())

    print(df_meta.to_string())

    # # Combine the results of all the ASR benchmarks and plot the results
    # df, plt, df_meta = nse.combined_asr_performance(benchmarks, participant="PAR")
    # plt.savefig("asr_out_PAR.png")
    # plt.clf()

    # # Combine the results of all the ASR benchmarks and plot the results
    # df, plt, df_meta = nse.combined_asr_performance(benchmarks, participant="INV")
    # plt.savefig("asr_out_INV.png")
    # plt.clf()
