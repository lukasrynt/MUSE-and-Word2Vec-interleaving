from src.orchestration import Orchestrator

en_words = ['access', 'authoris', 'capit', 'committe', 'consum', 'content', 'deadlin', 'document', 'expenditur',
            'full', 'geograph', 'group', 'growth', 'hear', 'instrument', 'investig', 'label', 'list', 'loan',
            'network', 'opinion', 'partnership', 'pension', 'price', 'sale', 'secret', 'solut', 'trade', 'type',
            'wast']
cz_words = ['přístup', 'povolen', 'kapitál', 'výbor', 'spotřebitel', 'obsah', 'lhůt', 'dokument', 'výdaj', 'pln',
            'zeměpisn', 'skup', 'růst', 'slyšen', 'nástroj', 'šetřen', 'označen', 'seznam', 'úvěr', 'síť',
            'stanovisk', 'partnerst', 'důchod', 'cen', 'prodej', 'tajemstv', 'řešen', 'obchod', 'typ', 'odpad']


def main():
    orch = Orchestrator(en_words=en_words, cz_words=cz_words,
                        muse_epoch_size=1, muse_epochs=5000,
                        vector_sizes=[300], window_sizes=[6])
    orch.run_all()


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
