"""Adversarial saved-record checks. These fixtures are synthetic, not GPU runs."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('hidden_audit',Path(__file__).resolve().parents[1]/
                                         'scripts/audit_prm_hidden_results.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class HiddenAuditTests(unittest.TestCase):
    def setUp(self):
        ids=[351,95,5478,3542,359,5031,0,6225]
        self.labels=dict(zip(ids,[2/3,0,0,0,2/3,0,1,1]))
        self.positions={t:[3] for t in ids}
        self.ref=[{'task_id':t,'turn_end_indices':[3],'step_scores':[.2],'total':.2} for t in ids]
        self.summary={'identity':{'source_commit':'synthetic'},'samples_sha256':m.SAMPLES_SHA,
                      'train_ids_sha256':m.TRAIN_SHA,'total_samples':8,'retained_samples':8,
                      'rejected':[],'optimizer_steps':2,'samples_consumed':16,
                      'optimization':{'gradient_accumulation_steps':8,'weight_decay':.01,
                                      'max_grad_norm':1.0},
                      'peak_allocated_gib':0,'peak_reserved_gib':0,'records':[]}
        for step in (1,2):
            self.summary['records'].append({'optimizer_step':step,'samples_consumed':step*8,
                'task_ids':ids,'terminal_rewards':list(self.labels.values()),'gradient_norm':2.,
                'loss':m.score(dict.fromkeys(ids,0.),self.labels)['mse'],
                'totals_before_update':[0.]*8,'gradient_clipped':True})
        self.reload={'predictions':copy.deepcopy(self.ref),'identity':{'source_commit':'synthetic'},
                     'reference_match':True,'rejected':[],'total_samples':8}

    def audit(self):
        return m.audit_run(self.summary,self.ref,self.reload,self.labels,self.positions)

    def test_complete_records_recompute_metrics_and_groups(self):
        result=self.audit()
        self.assertAlmostEqual(result['mse'],m.score(dict.fromkeys(self.labels,.2),self.labels)['mse'])
        self.assertEqual(result['segment_clip_ratio'],1)
        self.assertEqual(result['full_minus_zero_group_mean'],0)
        self.assertFalse(result['memory_gate_passed'])

    def test_saved_reference_and_reload_must_both_cover_every_task(self):
        self.reload['predictions'].pop()
        with self.assertRaises(ValueError): self.audit()

    def test_duplicate_task_cannot_replace_missing_task(self):
        self.ref[-1]=self.ref[0]
        with self.assertRaises(ValueError): self.audit()

    def test_nonfinite_reload_is_rejected_even_with_true_match_flag(self):
        self.reload['predictions'][0]['step_scores']=[float('nan')]
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_score_position_is_rejected(self):
        self.ref[0]['turn_end_indices']=[2]
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_clipping_flag_is_rejected(self):
        self.summary['records'][0]['gradient_clipped']=False
        with self.assertRaises(ValueError): self.audit()

    def test_changed_training_labels_are_rejected(self):
        self.summary['records'][0]['terminal_rewards']=[0]*8
        with self.assertRaises(ValueError): self.audit()

    def test_wrong_exposure_counter_is_rejected(self):
        self.summary['records'][0]['samples_consumed']=16
        with self.assertRaises(ValueError): self.audit()

    def test_missing_optimizer_step_is_rejected(self):
        self.summary['records'][1]['optimizer_step']=3
        with self.assertRaises(ValueError): self.audit()

    def test_recorded_loss_is_independently_checked(self):
        self.summary['records'][0]['loss']=42.
        with self.assertRaises(ValueError): self.audit()

    def test_resume_segment_does_not_claim_cumulative_clipping_coverage(self):
        for row,step in zip(self.summary['records'],(17,18)):
            row['optimizer_step']=step;row['samples_consumed']=step*8
        self.summary['optimizer_steps']=18;self.summary['samples_consumed']=144
        result=self.audit()
        self.assertEqual(result['segment_optimizer_steps'],2)
        self.assertEqual(result['cumulative_optimizer_steps'],18)
        self.assertEqual(result['segment_clip_count'],2)


if __name__=='__main__': unittest.main()
