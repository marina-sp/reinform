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
        self.test_rollouts = self.option.test_times

        if data is None and n_seq is None:
            raise AttributeError("One of 'data' or 'n_seq' should not be None.")

        if data is None:
            self.mode = "answer"
            self.prev_rel = torch.ones((n_seq, 1), dtype=torch.long) * self.vocab.start_token_id
            self.set_path(self.prev_rel)
            self.add_steps(self.query_ent)  # self.query_ent
        else:
            self.mode = "context"
            self.set_path(data)  # answers
            self.add_steps(self.query_rel, self.query_ent) #

    def is_first_step(self):
        if self.mode == "context" and self.steps == 3:
            return True
        elif self.mode == "answer" and self.steps == 2:
            return True
        else:
            return False

    def do_rollout(self, data):
        return data.repeat_interleave(self.option.test_times, 0)

    def get_current_ent(self, hide=False):
        # hide query entity for the agent
        if self.is_first_step():
            return torch.ones_like(self.query_ent) * self.vocab.unk_token_id
        return self.current_ent if not hide else self.hide_emerging(self.current_ent)

    def get_query_ent(self, do_rollout=False):
        return self.query_ent if not do_rollout else self.query_rel.repeat_interleave(self.option.test_times, 0)

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

    def add_steps(self, *steps):
        # print(steps)
        for step in steps:
            new_step = step.reshape(self.n, -1).cpu().type(torch.long)
            self.set_path(torch.cat([self.path, new_step], dim=1))

    def hide_emerging(self, data):
        out = data.detach().clone()
        emerging_mask = self.vocab.is_test_entity(out, strict=False)
        out[emerging_mask] = self.vocab.unk_token_id
        return out

    def get_context_path(self, step=None, test_times=1):
        """
        Manage path construction appropriate for evalution with a path-scoring model.
        h - r - t - context steps (h is the entity to be predicted)

        :param step: int: How many of the selected actions (steps) to include in the output path. 0 means only query
        :param test_times: int: the number of paths selected by beam for the same test query
        :return: torch.tensor: with paths for every query triple
        """
        if step is None:
            last_pos = self.steps
        else:
            # every step consists of two items in the path (rel and ent id)
            last_pos = step * 2 + (2 if self.mode=="answer" else 3)

        if self.mode == "context":
            full_path = self.path
        else:
            # this lines is reached only in the test setting
            # - save top 1
            sequences = self.get_answer_path().cpu()[::test_times]

            # - add reversed query to the path
            # t=mask rel_inv h=start_entities -- path
            inv_queries = torch.tensor([
                self.vocab.rel2inv[rel.item()] for rel in self.query_rel
            ])
            full_path = torch.cat((self.answer.view(-1, 1), inv_queries.view(-1, 1), sequences), -1)

        return full_path[:, :last_pos]

    def get_answer_path(self):
        assert self.mode == "answer"  # context evaluation of answer paths is impossible, sort it out
        return self.path[:, 1:]  # drop artificial prev rel

    def get_output_path(self, out_mode, test_times):
        # todo: fix masked query entity
        if (self.mode == "context") and (out_mode == "context"):
            # answer - qr - qe (masked) - context
            sequences = self.get_context_path().cpu()
            triples = self.path[:, :3]
        elif (self.mode == "answer") and (out_mode == "context"):
            # post-process sequences from Minerva for context evaluation
            sequences = self.get_context_path(test_times=test_times)
            triples = torch.stack((self.query_ent, self.query_rel, self.answer), dim=1)
        elif (self.mode == "answer") and (out_mode == "answer"):
            # sequences can be printed as is - but only the top 1
            sequences = self.get_answer_path().cpu()[::test_times]
            triples = torch.stack((self.query_ent, self.query_rel, self.answer), dim=-1)
        else:
            raise ValueError("Cannot train a model on context and evaluate on answer!")

        #if out_mode == "context":
        #    # unmask: answer - qr - qe (masked) - context
        #    assert (sequences[:, 2] == self.vocab.unk_token_id).all()
        #    sequences[:, 2] = self.query_ent
        #else:
        #    # unmask: qe (masked) - path
        #    # important: answer paths already dont have the rollouts
        #    assert (sequences[:, 0] == self.vocab.unk_token_id).all()
        #    sequences[:, 0] = self.query_ent
        return sequences, triples

    def get_action_space(self, step):
        # start entities never come with rollouts, but current entities do: roll out for
        assert len(self.current_ent) % len(self.query_ent) == 0
        if not self.is_first_step():
            query_ent = self.do_rollout(self.query_ent)
            query_rel = self.do_rollout(self.query_rel)
            answer = self.do_rollout(self.answer)
            all_correct = [x for x in self.all_correct for _ in range(self.test_rollouts)]
        else:
            query_ent = self.query_ent
            query_rel = self.query_rel
            answer = self.answer
            all_correct = self.all_correct

        action_space = self.graph.get_out(
            self.current_ent, query_ent, query_rel, answer, all_correct, step)

        # actions to emerging ents have been already padded in the graph
        assert (action_space == self.hide_emerging(action_space)).all()
        return action_space, self.hide_emerging(action_space)
