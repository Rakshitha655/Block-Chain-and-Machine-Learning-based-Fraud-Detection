# pbft_blockchain.py
import hashlib, json, time, uuid
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class Transaction:
    tx_id: str
    sender: str
    receiver: str
    amount: float
    timestamp: float
    features: dict
    ml_is_fraud: int = 0  # ML prediction 0/1

@dataclass
class Block:
    index: int
    prev_hash: str
    transactions: List[Transaction]
    timestamp: float
    proposer: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self):
        block_dict = {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "transactions": [asdict(t) for t in self.transactions],
            "timestamp": self.timestamp,
            "proposer": self.proposer,
            "nonce": self.nonce
        }
        return hashlib.sha256(json.dumps(block_dict, sort_keys=True).encode()).hexdigest()

class SimpleContractStorage:
    """Very small smart-contract like storage executed when blocks are committed."""
    def __init__(self):
        self.store = {}

    def store_transaction(self, tx: Transaction):
        self.store[tx.tx_id] = {
            "sender": tx.sender,
            "receiver": tx.receiver,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
            "ml_is_fraud": int(tx.ml_is_fraud)
        }

    def flag_fraud(self, tx_id: str, flag:int):
        if tx_id in self.store:
            self.store[tx_id]['ml_is_fraud'] = int(flag)
        else:
            self.store[tx_id] = {"ml_is_fraud": int(flag)}

    def query(self, tx_id: str):
        return self.store.get(tx_id, None)

class PBFTNode:
    def __init__(self, node_id, network):
        self.node_id = node_id
        self.network = network
        self.chain = []
        self.contract = SimpleContractStorage()
        genesis = Block(index=0, prev_hash="0", transactions=[], timestamp=time.time(), proposer="genesis")
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)

    def validate_block(self, block: Block):
        if block.prev_hash != self.chain[-1].hash:
            return False
        if block.compute_hash() != block.hash:
            return False
        return True

    def on_commit(self, block: Block, commit_signatures):
        self.chain.append(block)
        for tx in block.transactions:
            self.contract.store_transaction(tx)
        print(f"[Node {self.node_id}] Committed block {block.index} ({len(block.transactions)} txs).")

class PBFTNetwork:
    def __init__(self, n_nodes=4):
        assert n_nodes >= 4 and (n_nodes - 1) % 3 == 0, "Use n = 3f+1 (e.g., 4,7,10...)."
        self.n = n_nodes
        self.nodes = {i: PBFTNode(i, self) for i in range(n_nodes)}
        self.primary = 0
        self.f = (n_nodes - 1) // 3

    def broadcast_preprepare(self, block: Block):
        msgs = []
        for nid in self.nodes:
            msgs.append(("PRE-PREPARE", nid, block))
        return msgs

    def run_consensus_for_block(self, proposing_node:int, block: Block):
        preprepare_msgs = self.broadcast_preprepare(block)
        prepare_votes = {}
        for _, nid, blk in preprepare_msgs:
            node = self.nodes[nid]
            valid = (blk.prev_hash == node.chain[-1].hash) and (blk.compute_hash() == blk.hash)
            prepare_votes[nid] = valid
        prepares_yes = sum(1 for v in prepare_votes.values() if v)
        if prepares_yes < (2*self.f + 1):
            print("[Network] Prepare failed. prepares_yes:", prepares_yes)
            return False
        commit_signatures = [nid for nid,v in prepare_votes.items() if v]
        if len(commit_signatures) < (2*self.f + 1):
            print("[Network] Commit failed. commits:", len(commit_signatures))
            return False
        for nid, node in self.nodes.items():
            node.on_commit(block, commit_signatures)
        return True

    def propose_block_from_primary(self, transactions: List[Transaction]):
        primary_node = self.nodes[self.primary]
        new_index = primary_node.chain[-1].index + 1
        prev_hash = primary_node.chain[-1].hash
        block = Block(index=new_index, prev_hash=prev_hash, transactions=transactions, timestamp=time.time(), proposer=self.primary)
        block.hash = block.compute_hash()
        print(f"[Network] Primary {self.primary} proposed block #{new_index} hash={block.hash[:8]}")
        ok = self.run_consensus_for_block(self.primary, block)
        return ok

def make_transaction(sender, receiver, amount, features=None, ml_flag=0):
    return Transaction(
        tx_id = str(uuid.uuid4()),
        sender=sender,
        receiver=receiver,
        amount=amount,
        timestamp=time.time(),
        features=features or {},
        ml_is_fraud=int(ml_flag)
    )

if __name__ == "__main__":
    net = PBFTNetwork(n_nodes=4)  # 4 nodes -> f = 1
    txs = [
        make_transaction("Alice", "Shop1", 120.5, {"type":"online"}, ml_flag=0),
        make_transaction("Bob", "Shop2", 9000.0, {"type":"international"}, ml_flag=1),
    ]
    ok = net.propose_block_from_primary(txs)
    print("Consensus success?", ok)
    # query
    print("Query tx on node0:", net.nodes[0].contract.query(txs[1].tx_id))
