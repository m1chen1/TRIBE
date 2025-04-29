# Comments throughout... some apply to you, others were notes to myself in case I had to backtrack. Do read them.
# A useful call for this is: "python3 modelDecodePlay.py 2>&1 | tee 6_output_full.txt", this will 
# provide an output file for you to observe trends and help troubleshoot if necessary.  Be sure to do this.  
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.sdp_kernel = "flash"
import time
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
#from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
#from torchmetrics.classification import MulticlassAccuracy
from torch.nn.functional import scaled_dot_product_attention

class FlashSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x, key_padding_mask=None):
        # x: [B, L, E]
        #print("X is shaped as: ", x.shape)
        B, L, E = x.shape

        # project and reshape to (B, num_heads, L, head_dim)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # this will invoke the flash‐SDP kernel under the hood
        attn_out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout,
            is_causal=False
        )  # → [B, num_heads, L, head_dim]

        # reshape back and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, E)
        return self.out_proj(attn_out)

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = FlashSelfAttention(d_model, nhead, dropout)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout1  = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.dropout2  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        # self‑attention
        sa = self.self_attn(src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(sa)
        src = self.norm1(src)
        # feed‑forward
        ff = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src


class CustomPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=int(1024*1024*0.2)):
        super(CustomPositionalEncoding, self).__init__()
        
        # Initialize a learnable positional encoding matrix
        self.encoding = nn.Parameter(torch.zeros(max_len, embed_dim))
        nn.init.xavier_uniform_(self.encoding)  # Xavier initialization for better training stability

    def forward(self, x):
        # Add the learnable positional encoding to the input tensor
        return x + self.encoding[:x.size(1), :]

# path: /home/TRIBE_SHARE/MalwareBazaar/stripped_malware
class BinaryFileDataset(Dataset):
    def __init__(self, file_paths, max_file_size=int(0.2*1024*1024)):#10 * 1024 * 1024):
        self.file_paths = file_paths
        self.max_file_size = max_file_size
    
    def __len__(self):
        return len(self.file_paths)

    #MODIFY 
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as file:
            byte_data = np.frombuffer(file.read(), dtype=np.uint8)
            file_size = len(byte_data)
            if file_size > self.max_file_size:
                byte_data = byte_data[:self.max_file_size]  # Truncate if over the size limit
                print(f"Truncated file: {file_path} to {self.max_file_size} bytes")
            else:
                print(f"Loading file: {file_path}, Size: {file_size} bytes")
            return torch.tensor(byte_data, dtype=torch.long)


# Collate function for padding and mask generation
def collate_fn(batch):
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    num_masks = 9 
    idxs = [[l] + [random.randint(1, l-1) for p in range(0, num_masks-2)]for l in lengths]
    #print(f"Batch max sequence length: {max_len}")

    # No separate attention mask for the encoder is necessary as long as the batch size remains 1, else must
    # create a separate attention mask.
    padded_batch = torch.zeros((len(batch),1, max_len), dtype=torch.long)
    attention_masks = torch.ones((len(batch),num_masks, max_len), dtype=torch.bool) 
    labels = torch.zeros((len(batch), num_masks), dtype=torch.long)

    #encoder expects a key padding mask where position to ignore are marked True, so ones.
    
    for i, seq in enumerate(batch):
        labels[i, 0] = seq[0]
        padded_batch[i, :, :len(seq)] = seq
        for j, k in enumerate(idxs[i]):
            #padded_batch[i, j+1, :k] = seq[:k]  # Copy actual sequence up to k, initial is the completely masked input
            attention_masks[i, j+1, :k] = 0  # Mask valid tokens up to k: also changed to behavior expected by transformer. as described below.
            labels[i, j+1] = seq[k-1]

    # Reshape to conform new batching
    padded_batch = torch.reshape(padded_batch, [-1, max_len])
    attention_masks = torch.reshape(attention_masks, [-1, max_len])
    labels = torch.reshape(labels, [-1])

    # Reshape to super batching
    labels2 = labels.reshape(len(batch),3,-1).squeeze(0)
    attention_masks2 = attention_masks.reshape(len(batch), 3, 3, -1).squeeze(0)

    # The encoder expects a key padding mask where positions to ignore are marked as True. This is the opposite of what I had so I altered to reflect.
    #padded_batch = nn.functional.one_hot(padded_batch,num_classes=256)
    attention_masks2 = attention_masks2.to(torch.bool)
    return padded_batch, attention_masks2, labels2

# Transformer Autoencoder Model
class TransformerAutoencoder(nn.Module):
    def __init__(self, emb_size=256, n_heads=4, num_layers=4, latent_dim=64, vocab_size=256):
        super(TransformerAutoencoder, self).__init__()
        
        # Encoder
        #encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, batch_first=True)
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self.positional_encoder_layer = CustomPositionalEncoding(embed_dim=emb_size)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_encode = nn.Linear(emb_size, latent_dim)
        self.encoder_layers = nn.ModuleList(
            [FlashTransformerEncoderLayer(emb_size, n_heads, dim_feedforward=emb_size*4, dropout=0.1)
            for _ in range(num_layers)]
        )

        # # Decoder
        # self.fc_decode = nn.Linear(latent_dim, emb_size)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=n_heads, batch_first=True)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # self.output_layer = nn.Linear(emb_size, vocab_size)  # Predict original tokens

        # Classifier
        # self.positional_decoder_layer = CustomPositionalEncoding(embed_dim=emb_size)
        # self.fc_decode = nn.Linear(latent_dim, emb_size)
        # decoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_heads, batch_first=True)
        # self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        # self.output_layer = nn.Linear(emb_size, vocab_size)  # Predict original tokens
        self.positional_decoder_layer = CustomPositionalEncoding(embed_dim=emb_size)
        self.fc_decode = nn.Linear(latent_dim, emb_size)
        # build a list of flash‑attention decoder layers
        self.decoder_layers = nn.ModuleList([
            FlashTransformerEncoderLayer(
                d_model=emb_size,
                nhead=n_heads,
                dim_feedforward=emb_size*4,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        # final output projection back to token logits
        self.output_layer = nn.Linear(emb_size, vocab_size)
    
    def encoder(self, x): # attention_mask):
        x = self.positional_encoder_layer(x)
        #encoded = self.transformer_encoder(x) # was the logical not of the mask... making it opposite in the dataloader
        # sequentially apply each flashy layer
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=None)
        return self.fc_encode(x.mean(dim=1))

    # Use this method to get encodings.
    def get_encodings(self, x):
        x = self.embedding(x)
        lat_vec = self.encoder(x)
        return lat_vec

    # # Second Decoder Attempt Efficient change to using broadcasting vice .repeat() for memory efficiency.
    # def decoder(self, latent_vector, x, attention_mask):
    #     batch_repeat = attention_mask.shape[0]

    #     # Efficiently broadcast decoded_features without repeat
    #     decoded_features = self.fc_decode(latent_vector).unsqueeze(1)  # [1, 1, emb_size]
    #     decoded_features = decoded_features.expand(batch_repeat, -1, -1)  # Efficient broadcast

    #     # Broadcast x similarly without explicit repeat
    #     x = self.positional_decoder_layer(x)
    #     x = x.expand(batch_repeat, -1, -1)

    #     # Concatenate features and inputs efficiently
    #     thing = torch.cat([decoded_features, x], dim=1)

    #     # Attention mask handling
    #     mask = torch.zeros(batch_repeat, 1, device=x.device, dtype=torch.bool)
    #     mask = torch.cat([mask, attention_mask], dim=1)

    #     reconstructed = self.transformer_decoder(src=thing, src_key_padding_mask=mask)
    #     decode = self.output_layer(reconstructed.mean(dim=1))

    #     return decode

    #Third Decoder Attempt leveraging torch flash attention

    def decoder(self, latent_vector, x_embed, attention_mask):
        """
        latent_vector:   [batch, latent_dim]
        x_emb:           [batch, seq_len, emb_size]  (already embedded)
        attention_mask:  [batch, seq_len]  (bool mask—True==pad)
        """

        # 1) project latent → embedding, and prep token embeddings + pos‑enc
        #print("x_embed is: ", x_embed.shape)
        #x = x.squeeze()
        N, L = attention_mask.shape
        E = x_embed.size(2)
        x = x_embed.expand(N, L, E)
        latV = latent_vector.expand(N, -1)

        feat = self.fc_decode(latV)        # → [B, emb_size]
        feat = feat.unsqueeze(1).expand(N, 1, E)   # → [B, 1, emb_size]

        # Use already embedded inputs
        x=self.positional_decoder_layer(x)

        # 2) concatenate latent prepended to token stream
        #    result: [B, L+1, emb_size]
        inp = torch.cat([feat, x], dim=1)

        # 3) run through each FlashTransformerEncoderLayer
        #    passing through the same key-padding mask (prepended with a False for the latent token)
        #    src_key_padding_mask: True==pad positions
        #    so we need to prepend a “not-pad” for the latent token
        pad_mask = attention_mask                    # [B, L]
        prepended = torch.zeros((N, 1), dtype=torch.bool, device=pad_mask.device)
        src_key_padding_mask = torch.cat([prepended, pad_mask], dim=1)  # [B, L+1]

        out = inp
        for layer in self.decoder_layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask)

        # 4) pool & final projection
        #    you could also project each position and compute cross‑entropy per token,
        #    but given your original code you take the mean over time:
        pooled = out.mean(dim=1)    # [B, emb_size]
        return self.output_layer(pooled)


    # def decoder(self, latent_vector, x, attention_mask):
    #     # Decode
    #     #print("The latent dimension is: ", latent_vector.shape)
    #     x = x.repeat(attention_mask.shape[0], 1, 1)
    #     decoded_features = self.fc_decode(latent_vector).unsqueeze(1).repeat(attention_mask.shape[0], 1, 1)
    #     thing = torch.cat([decoded_features, x], 1)
    #     #print("Shape of thing is: ", thing.shape)
    #     mask = torch.zeros([attention_mask.shape[0], 1]).to(device)
    #     mask = torch.cat([mask, attention_mask], 1)
    #     #print("Shape of mask is: ", mask.shape)
    #     #latent_vector = latent_vector.unsqueeze(1).repeat(1, m.shape[1], 1)
    #     #ones_vector = torch.ones_like(decoded_features)
    #     reconstructed = self.transformer_decoder(src=thing, src_key_padding_mask = mask) # was the logical not of the mask... trying without # was the full attention_mask
    #     decode = self.output_layer(reconstructed.mean(dim=1)) # took out softmax as CrossEntropyLoss in torch expects logits and does the softmax for you.
    #     return decode

    def forward(self, token_ids, attention_mask):
        x_embed = self.embedding(token_ids)
        lat_vec = self.encoder(x_embed)
        output = self.decoder(lat_vec, x_embed, attention_mask)
        return output

scaler = GradScaler('cuda')

# Training Function
def train(model, dataloader, optimizer, criterion, device, epoch, effec_batch):
    model.train()
    total_loss = 0.0
    backs = 0
    #optimizer.lr/=float(maxbacks)
    best_batch_loss = 1000
    per100_accuracy = 0
    last_100_accuracy = 'unk'
    outs = torch.zeros((9,256), dtype=torch.float16, device=device)

    for batch_idx, (batch, super_mask, super_labels) in enumerate(dataloader):
        batch = batch.to(device)
        if backs == 0:
            optimizer.zero_grad()
        ctr = 0
        outs.zero_()
        for mask, labels in zip(super_mask,super_labels):
            mask, labels = mask.to(device), labels.to(device)
            with autocast('cuda'):
                output = model(batch, mask)
                #print("output shape = ", output.shape)
                loss = criterion(output.reshape(-1, 256), labels)  # Compare reconstructed output to input also replaced torch.squeeze(batch[:,1,:].view(-1)))
            outs[ctr*3:ctr*3+3] = output 

            scaler.scale(loss).backward()
            backs+=1
            # This logic implements collecting logits over effec_batch (#) malware samples before implementing the update.  This will make your learning much less variable and make updates 
            # smoother. If you would like to collect over more than effec_batch, change effec_batch to be higher.
            if backs == effec_batch:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                backs = 0
                global global_update_count
                global_update_count+=1
                scheduler.step(global_update_count)
                print("Model Update Completed:")
            total_loss += loss.item()
            print(f"Batch {batch_idx+1}/{ctr+1}/{len(dataloader)}, Loss: {loss.item():.4f}", flush=True)
            ctr+=1
            del mask, output  # Free memory
        with torch.no_grad():
            #print(f"output: {outs.reshape(-1, 256).shape}, truths: {super_labels.reshape(-1).shape}")
            #outs_fp32 = outs.float()
            preds = outs.argmax(dim=1)          # [9]
            truth = super_labels.reshape(-1).to(device)    # [9]
            per_file_accuracy = (preds == truth).float().mean().item()

            per100_accuracy += per_file_accuracy
            print("Per file accuracy: ", per_file_accuracy)
        if (batch_idx % 100 == 0):
            with torch.no_grad():
                last_100_accuracy = per100_accuracy/(100) # 3 accounts for number of batches in superbatch.
                print("Per byte accuracy on last 100 files: ", last_100_accuracy)
                per100_accuracy = 0
        # this current saving mechanism just uses the training set. This is because you have not built a dataloader object for a validation
        # set. In order to implement what we discussed (to avoid overfitting with a validation loss tracking) you would need to do this, 
        # but it may be computationally expensive. Perhaps it would be good enough to implement a very small validation set. Ask if you want to 
        # discuss trade offs. alternatively, you can leave it as it is.
        if batch_idx%10==0: 
            # set at 10 for testing with binary set. Recommend at least 1000 for malware.
            temp_loss = total_loss/(len(batch)*3*(batch_idx+1))
            if (batch_idx <= 20):
                print(f"Conventional latest model save, temporary loss now: {temp_loss}")
                torch.save(model, './models/9testModels/10_Corrected_TRIBEmodel_latest_not_best')
                #torch.save(model.state_dict(), './models/testmodels/2_Corrected_TRIBEmodel_latest_not_best2')
            elif temp_loss < best_batch_loss:
                best_batch_loss = temp_loss
                torch.save(model, './models/9testModels/' + '10_Corrected_TRIBEmodel_loss_' + str(temp_loss) + 'last100acc' + str(last_100_accuracy))
                #torch.save(model.state_dict(), './models/testmodels/' + '2_Corrected_TRIBEmodel_loss_' + str(temp_loss) + 'last100acc' + str(last_100_accuracy))
                print("Model saved: ", epoch, "_TRIBEmodel_loss = ", temp_loss)
            else:
                print(f"Conventional latest model save, temporary loss now: {temp_loss}")
                torch.save(model, './models/9testModels/10_Corrected_TRIBEmodel_latest_not_best')
                #torch.save(model.state_dict(), './models/testmodels/2_Corrected_TRIBEmodel_latest_not_best2')
        #del mask, output  # Free memory
        del batch, super_mask, super_labels
    del outs
    torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

# Setup Training Pipeline
if __name__ == "__main__":

# Get first 100 files from the directory
    directory = "/home/TRIBE_SHARE/MalwareBaazar/stripped_malware/training_data"
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]#[:50]

    batch_size = 1
    dataset = BinaryFileDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    #metric = MulticlassAccuracy(num_classes = 256).to('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # If Loading existing trained model for further training
    #model = torch.load("./models/testmodels/0_TRIBEmodel_loss_3.6755192106346994").to(device)
    #model = torch.load("./models/6testModels/6_Corrected_TRIBEmodel_loss_4.8728894655693304last100acc0.001111111119389534").to(device)
    model = torch.load("./models/9testModels/10_Corrected_TRIBEmodel_latest_not_best").to(device)
    #  else, 
    #model = TransformerAutoencoder().to(device)
    # since we cannot fit multiple malware in memory simultaneously along with the model, we implement effective batching 
    # by accumulating gradients that many times before updating see the training function.
    # but keep LayerNorm (and any other norm layers) in fp32 for stability
    effec_batch = 15 # 3 batches per superbatch and 5 superbatches per effective batch, all counted
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    #max_updates = 3,000
    global_update_count = 0
    scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=100, T_mult=2, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()#nn.CrossEntropyLoss()#nn.NLLLoss()
    best_train_loss = 1000

    # Maybe validation loss check with saves is best implemented here
    for epoch in range(100):
        print(f"Epoch {epoch+1} starting...")
        loss = train(model, dataloader, optimizer, criterion, device, epoch=epoch, effec_batch=effec_batch)
        print(f"Epoch {epoch+1} completed, Loss: {loss:.4f}")

        # if loss < best_train_loss:
        #     best_train_loss = loss
        #     torch.save(model, './models/6testModels/' + str(epoch+1) + 'Cor_TRIBE_model_' + str(loss))
        #     print("Model saved: ", (epoch+1), "TRIBEmodel ", "loss = ", loss)
            
