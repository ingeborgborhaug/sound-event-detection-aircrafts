def plot_waveforms(signals, sr, title_prefix= "", subfolder=""):
    time = np.arange(signals.shape[0]) / sr
    plt.figure(figsize=(14,8))
    for i in range(signals.shape[1]):
        plt.subplot(signals.shape[1], 1, i+1)
        plt.plot(time, signals[:, i])
        #plt.title(f"{title_prefix} {i+1} - Waveform")
        plt.xlabel("Time [s]", labelpad=15)
        plt.ylabel("Amplitude", labelpad=15)
    plt.tight_layout()
    name = title_prefix.replace(' ', '_')
    save_path = f'dataset/synthetic_data/figures/{subfolder}waveform_{name}'
    plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f'Figure saved to {save_path}')