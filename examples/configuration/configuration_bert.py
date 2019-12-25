
from transformers import BertConfig

class BertForDependencyParsingConfig(BertConfig):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_postag=False,
        num_postags=0,
        num_labels=0,
        arc_space=512,
        label_space=128,
        **kwargs
    ):
        super(BertForDependencyParsingConfig, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )
        self.use_postag = use_postag
        self.num_postags = num_postags
        self.num_labels = num_labels
        self.arc_space = arc_space
        self.label_space = label_space


class BertForDependencyParsingWithOrderConfig(BertConfig):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_postag=False,
        num_postags=0,
        num_labels=0,
        arc_space=512,
        label_space=128,
        max_parsing_order=32,
        order_space=128,
        **kwargs
    ):
        super(BertForDependencyParsingWithOrderConfig, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            **kwargs
        )
        self.use_postag = use_postag
        self.num_postags = num_postags
        self.num_labels = num_labels
        self.arc_space = arc_space
        self.label_space = label_space
        self.max_parsing_order = max_parsing_order
        self.order_space = order_space