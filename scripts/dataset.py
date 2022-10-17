import torch


# currently not in use, was used in a few experimental stages of this project

class Dataset_from_DF:
    """
    Helper class to create dataset for pytorch data loader
    """

    def __init__(self, dataframe, tokenizer, max_length, device, text_column="text"):
        self.len = len(dataframe)
        self.data = dataframe.reset_index()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.text_column = text_column

    def __getitem__(self, index):
        text = str(self.data[self.text_column][index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus("het" + text[:10_000],  # temp fix to not create empty tensor lists.
                                            None, add_special_tokens=False,  # max_length=self.max_length,
                                            # padding='max_length',
                                            # return_token_type_ids=True,
                                            # truncation=True,
                                            # return_overflowing_tokens=True,
                                            # num_truncated_tokens=True
                                            )

        input_ids = inputs["input_ids"]
        i = 0
        input_idss = []
        attention_idss = []

        for j in range(0, len(input_ids) + 512, 512):
            if i == j: continue
            arr = input_ids[i:j]

            if len(arr) != 512:
                m = torch.zeros(512, dtype=torch.long)
                att = torch.zeros(512, dtype=torch.long)
                l_arr = len(arr)

                m[:l_arr] = torch.tensor(arr, dtype=torch.long)
                att[:l_arr] = torch.ones(l_arr, dtype=torch.long)

                input_idss.append(m)
                attention_idss.append(att)

            else:
                input_idss.append(torch.tensor(arr, dtype=torch.long))
                attention_idss.append(torch.ones(512, dtype=torch.long))
            i = j

        input_ids_processed = torch.stack(input_idss)
        attention_ids_processed = torch.stack(attention_idss)

        return {"input_ids": input_ids_processed, "attention_mask": attention_ids_processed}

    def __len__(self):
        return self.len
