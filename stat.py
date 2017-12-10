"""
Part of BME595 project
Program:
  Show statistics of dataset
"""
from collections import Counter
from data import data_loader, _preprocess_dataset_small, _preprocess_dataset_large


def show_distribution(max_len=60, deduplicate=False):
    small_sentences, small_polarities, purposes, _ = _preprocess_dataset_small(max_len, deduplicate=deduplicate)
    large_sentences, large_polarities, polarity_to_idx = _preprocess_dataset_large(max_len, deduplicate=deduplicate)
    purpose_size = len(small_sentences)
    polarity_size = len(small_sentences) + len(large_sentences)
    print('\nsmall dataset size:', len(small_sentences))
    print('large dataset size:', len(large_sentences))
    print('purpose data size:', purpose_size)
    print('polarity data size (merge small and large):', polarity_size)

    print('\npurpose distribution:')
    purpose_to_idx = {'Criticizing': 0, 'Comparison': 1, 'Use': 2,
                      'Substantiating': 3, 'Basis': 4, 'Neutral': 5}
    ctr = Counter(purposes)
    for purpose, idx in purpose_to_idx.items():
        print(purpose.ljust(30), ctr[idx]/purpose_size)

    print('\npolarity distribution:')
    polarity_to_idx = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
    ctr = Counter(small_polarities+large_polarities)
    for polarity, idx in polarity_to_idx.items():
        print(polarity.ljust(30), ctr[idx]/polarity_size)

if __name__ == '__main__':
    show_distribution()


