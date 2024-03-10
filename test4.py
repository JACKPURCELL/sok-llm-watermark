from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large").cuda()


# Now, process some English text as well
text_inputs = processor(text="Hello, my dog is cute", src_lang="eng", return_tensors="pt").cuda()

# from text
output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)
translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(translated_text_from_text)