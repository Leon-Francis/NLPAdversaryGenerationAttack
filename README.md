# NLPAdversaryGenerationAttack

## experiment record

> 2021-4-29 -(bert_encode)> (hidden + noise)  -(gan_gen)> adv_hidden -(LSTM_decode)> adv_logits
>
> generate is trained by loss_Similarity by discrimitor and loss_Attack by target model 
>
> didn't work, can not generate normal sentence.
```
------orginal sentence---------
even the trailer for this movie makes me cry , like the first time i saw this movie . not for people who are easily upset by intense material ! the finest performances by alan rickman and madelaine stowe , without a doubt . this dreadful tale of a society with the power to kidnap and torture it ' s citizens for any reason , whether they are anarchist ' s or the writer of children ' s books will chill you to the bone . i saw it when it first came out 1991 and i remember every frame . it still scares the hell out me today . it ' s happening now . < br / > < br / > apparently , imdb requires ten lines to meet their criteria for a film review . imdb might want to get a grip ! some of us are a little more succinct about writing opinions . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
------setence -> encoder -> decoder-------
even the trailer for this movie makes me cry , like the first time i saw this movie . not for people who are easily upset by intense material ! the finest performances by alan rickman and maderitae stowe , without a doubt . this dreadful tale of a society with the power to kidnap and torture it ' s citizens for any reason , whether they are karate ' s or the writer of children ' s books will chill you to the bone . i saw it when it first came out 1991 and i remember every frame . it still scares the hell out me today . it ' s happening now . < br / > < br / > apparently , imdb requires ten lines to meet their criteria for a film review . imdb might want to get a grip ! some of us are a little more succinct about writing opinions . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
------sentence -> encoder  -> generator -> decoder-------
' c c c c c c c b b f f [SEP] [SEP] [SEP] ' ' ' hadn hadn hadn hadn hadn hadn hadn hadn didn didn didn didn weren weren weren weren weren weren weren weren weren weren weren weren weren doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt doubt decide decide decide [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn hadn weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren weren
```