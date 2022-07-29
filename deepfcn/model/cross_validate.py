def train_test_split(dataset):
    num_examples_dropped = 0
    if BALANCED:
        # TODO: Handle possibility that not all subjects will have the same # of bootstrapped examples (e.g. as a result from dropping examples in previous functions)
        num_pos, num_neg = len(dataset["neurodivergent"]), len(dataset["neurotypical"])
        cutoff = min({num_pos, num_neg})

        dataset["neurodivergent"] = dataset["neurodivergent"][:cutoff]
        dataset["neurotypical"] = dataset["neurotypical"][:cutoff]

        num_examples_dropped = abs(num_pos - num_neg)

    def train_test_split_helper(examples, label):
        num_examples, subject_to_examples = 0, defaultdict(list)
        for example in examples:
            num_examples += 1
            example["label"] = label
            subject_to_examples[example["subject_id"]].append(example)

        train_split_index = int(num_examples * (1 - TEST_SIZE - VALID_SIZE))
        valid_split_index = int(num_examples * (1 - TEST_SIZE))

        subject_ids = list(subject_to_examples.keys())
        random.shuffle(subject_ids)

        num_examples_counter = 0
        train_set, valid_set, test_set = [], [], []
        for subject_id in subject_ids:
            subject_examples = subject_to_examples[subject_id]
            num_subj_examples = len(subject_examples)
            max_overflow = int(num_subj_examples / 2) if num_subj_examples > 1 else 0

            if num_examples_counter + max_overflow <= train_split_index:
                train_set.extend(subject_examples)
            elif num_examples_counter + max_overflow <= valid_split_index:
                valid_set.extend(subject_examples)
            else:
                test_set.extend(subject_examples)

            num_examples_counter += len(subject_examples)

        return train_set, valid_set, test_set

    pos_train_set, pos_valid_set, pos_test_set = train_test_split_helper(
        dataset["neurodivergent"],
        label=[0.0, 1.0]
    )
    neg_train_set, neg_valid_set, neg_test_set = train_test_split_helper(
        dataset["neurotypical"],
        label=[1.0, 0.0]
    )

    valid_set = pos_valid_set + neg_valid_set
    test_set = pos_test_set + neg_test_set

    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    return train_set, valid_set, test_set, num_examples_dropped

def get_error_type(pred, label):
    get_class = lambda l: list(l).index(max(l))
    pred_class, label_class = get_class(pred), get_class(label)

    if not pred_class:
        if pred_class == label_class:
            return "tns" # True Negatives
        else:
            return "fns" # False Negatives
    else:
        if pred_class == label_class:
            return "tps" # True Positives
        else:
            return "fps" # False Positives


def calculate_accuracy(results):
    num_correct = results["tps"] + results["tns"]
    return num_correct / (num_correct + results["fps"] + results["fns"])


def calculate_precision(results):
    total = results["tps"] + results["fps"]
    return results["tps"] / total if total else 0.0


def calculate_recall(results):
    total = results["tps"] + results["fns"]
    return results["tps"] / total if total else 0.0


def train(gnn, train_set, optimizer, epoch):
    gnn.train()

    # Reduces learning rate every 50 epochs
    if not epoch % 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= (0.993 ** epoch)

    losses, error_metrics = [], {"tps": 0, "fps": 0, "tns": 0, "fns": 0}
    for example in train_set:
        optimizer.zero_grad()

        # Forward Pass
        y_true = torch.tensor(example["label"]).to(DEVICE)
        if NUM_COMMUNITIES == 0:
            y_pred = gnn(graph_to_data_obj(example))
        else:
            y_pred = gnn(example)

        loss = binary_cross_entropy(y_pred, y_true)

        # Backward pass
        loss.backward()
        optimizer.step()

        error_type = get_error_type(y_pred, y_true)
        error_metrics[error_type] += 1
        losses.append(loss.item())

    return mean(losses), calculate_accuracy(error_metrics)


def test(gnn, test_set):
    gnn.eval()

    error_metrics = {"tps": 0, "fps": 0, "tns": 0, "fns": 0}
    for example in test_set:
        y_true = torch.tensor(example["label"]).to(DEVICE)
        if NUM_COMMUNITIES == 0:
            y_pred = gnn(graph_to_data_obj(example))
        else:
            y_pred = gnn(example)

        error_type = get_error_type(y_pred, y_true)
        error_metrics[error_type] += 1

    return (
        calculate_accuracy(error_metrics),
        calculate_precision(error_metrics),
        calculate_recall(error_metrics)
    )

def cross_validate(examples, gnn, k=5, lr=1e-3, epochs=100, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = gnn.to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    # TODO: Batch learning (and then batch normalization)



def train_and_test(train_set, test_set, gnn):
    gnn = gnn.to(DEVICE)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=LR)

    # TODO: Batch learning (and then batch normalization)

    test_accuracies = []
    for epoch in range(1, NUM_EPOCHS + 1):
        mean_loss, train_acc = train(gnn, train_set, optimizer, epoch)
        test_acc, test_prec, test_recall = test(gnn, test_set)

        log_statement = ', '.join([
            f"\t\tEpoch: {epoch}",
            f"Loss: {round(mean_loss, 3)}",
            f"Train Acc: {round(train_acc * 100, 2)}",
            f"Test Acc: {round(test_acc * 100, 2)}",
            f"Test Prec: {round(test_prec * 100, 2)}",
            f"Test Recall: {round(test_recall * 100, 2)}",
        ])
        print(log_statement)

        test_accuracies.append(test_acc)

    return mean(test_accuracies[-15:])
