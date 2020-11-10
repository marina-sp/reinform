import torch

class State:
    # todo: add sequence writing from Trainer
    # todo: add sequence post processing (context-answer) interaction from Trainer

    def __init__(self, option, vocab, graph, query_rel, query_ent, answers, all_correct,
                 data=None, n_seq=None, prev_rel=None):
        """
         if data given --> context format
         if n_seq      --> answer (minerva) format
        :param vocab:
        :param data:
        :param n_seq:
        :param prev_rel:
        :param query_rel:
        :param query_ent:
        """
        self.option = option
        self.vocab = vocab
        self.graph = graph

        # sequence properties
        self.path = None
        self.n = 0
        self.steps = 0

        # state info
        self.prev_rel = prev_rel
        self.current_ent = None
        self.query_rel = query_rel
        self.query_ent = query_ent

        # episode info
        self.answer = answers
        self.all_correct = all_correct

        if data is None and n_seq is None:
            raise AttributeError("One of 'data' or 'n_seq' should not be None.")

        if data is None:
            self.mode = "answer"
            self.prev_rel = torch.ones((n_seq, 1), dtype=torch.long) * self.vocab.start_token_id
            self.set_path(self.prev_rel)
            self.add_steps(self.query_ent)
        else:
            self.mode = "context"
            self.set_path(data)  # answers
            self.add_steps(self.query_rel, self.query_ent)

    def get_current_ent(self, hide=False):
        return self.current_ent if not hide else self.hide_emerging(self.current_ent)

    def get_query_rel(self, do_rollout=False):
        return self.query_rel if not do_rollout else self.query_rel.repeat_interleave(self.option.test_times, 0)

    def get_answer(self, do_rollout=False):
        return self.answer if not do_rollout else self.answer.repeat_interleave(self.option.test_times, 0)

    def get_prev_rel(self):
        return self.prev_rel

    def set_path(self, data):
        data = data.type(torch.long)
        if len(data.shape) < 2:
            data = data.reshape(-1, 1)

        self.path = data
        self.n = self.path.shape[0]
        self.steps = self.path.shape[1]

        if self.steps > 1:
            self.current_ent = self.path[:, -1]
            self.prev_rel = self.path[:, -2]

    def get_context_path(self, test_times=1):
        if self.mode == "context":
            return self.path
        else:
            # this lines is reached only in the test setting
            # if there is a reward vs metric mismatch

            sequences = self.path[::test_times, 1:]  # drop artificial prev rel
            # - add reversed query to the path
            # t=mask rel_inv h=start_entities -- path
            inv_queries = torch.tensor([
                self.vocab.rel2inv[rel.item()] for rel in self.query_rel
            ])
            return torch.cat((self.answer.view(-1, 1), inv_queries.view(-1, 1), sequences), -1)

    def get_answer_path(self):
        assert self.mode == "answer"  # context evaluation of answer paths is impossible, sort it out
        return self.path[:, 1:]

    def get_eval_path(self, out_mode, test_times):
        if (self.mode == "context") and (out_mode == "context"):
            sequences = self.get_context_path().cpu()
            # triples = torch.stack((answers, queries, start_entities), dim=-1)
            triples = self.path[:, :3]
        elif (self.mode == "answer") and (out_mode == "context"):
            # post-process sequences from Minerva for context evaluation
            # - save top 1
            sequences = self.get_context_path(test_times).cpu()
            triples = torch.stack((self.query_ent, self.query_rel, self.answer), dim=1)
        elif (self.mode == "answer") and (out_mode == "answer"):
            # sequences can be printed as is - but only the top 1
            sequences = self.get_answer_path().cpu()[::test_times]
            triples = torch.stack((self.query_ent, self.query_rel, self.answer), dim=-1)
        else:
            raise ValueError("Cannot train a model on context and evaluate on answer!")
        return sequences, triples

    def get_action_space(self, step):
        action_space = self.graph.get_out(
            self.current_ent, self.query_ent, self.query_rel, self.answer,
            self.all_correct, step)
        return action_space, self.hide_emerging(action_space)

    def add_steps(self, *steps):
        #print(steps)
        for step in steps:
            new_step = step.reshape(self.n, -1).cpu().type(torch.long)
            self.set_path(torch.cat([self.path, new_step], dim=1))

    def hide_emerging(self, data):
        out = data.detach().clone()
        emerging_mask = self.vocab.is_test_entity(out, strict=False)
        out[emerging_mask] = self.vocab.unk_token_id
        return out
