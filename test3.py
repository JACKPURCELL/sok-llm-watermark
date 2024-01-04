# from transformers import AutoTokenizer,LlamaTokenizer

# tokenize = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')


# print(tokenize('hello world'))
# # print(tokenize('hello wordl'))
# # print(tokenize('hello word l'))
# print(tokenize('hello preferences'))
# print(tokenize.decode([2063]))

def translate_process(translation_queue, langs):
    import os
    import argostranslate.package
    import argostranslate.settings
    import argostranslate.translate

    # if device != "cpu":
    os.environ["ARGOS_DEVICE_TYPE"] = "cuda"
    argostranslate.settings.device = "cuda"
    # else:
    #     os.environ["ARGOS_DEVICE_TYPE"] = "cpu"
    #     argostranslate.settings.device = "cpu"

    # Install translation models
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    def install_model(la, lb):
        package_to_install = next(
            filter(
                lambda x: (x.from_code == la and x.to_code == lb),
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

    for i, li in enumerate(langs):
        for j, lj in enumerate(langs):
            if i == j:
                continue
            try:
                install_model(li, lj)
            except Exception:
                pass

    # Get actual models
    pairs = {}

    while True:
        task = translation_queue.get(block=True)
        if task is None:
            return

        text, la, lb, dst_queue = task
        if (la, lb) not in pairs:
            pairs[
                (la, lb)
            ] = argostranslate.translate.get_translation_from_codes(la, lb)

        try:
            dst_queue.put(pairs[(la, lb)].translate(text))
        except RuntimeError as e:
            print(e)
            print("Reducing number of CUDA threads")
            translation_queue.put(task)
            return

if __name__ == "__main__":
    translate_process(None, ["en", "fr"])