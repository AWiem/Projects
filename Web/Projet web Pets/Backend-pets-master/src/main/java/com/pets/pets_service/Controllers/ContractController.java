package com.pets.pets_service.Controllers;

import com.pets.pets_service.Exception.ResourceNotFoundException;
import com.pets.pets_service.Models.Contract;
import com.pets.pets_service.Repositories.ContractRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:4200")
@RestController
@RequestMapping("api/v1")
public class ContractController {
    @Autowired
    private ContractRepo contractRepo;

    @GetMapping("/contracts")
    public List<Contract> getAllContracts() {
        return contractRepo.findAll();
    }
    @GetMapping("/contracts/{id}")
    public ResponseEntity<Contract> getContractById(@PathVariable(value = "id") Integer contractId)
            throws ResourceNotFoundException {

        Contract contract = contractRepo.findById(contractId)
                .orElseThrow(() -> new ResourceNotFoundException("Contract not found"));
        return ResponseEntity.ok().body(contract);
    }

    @PostMapping("/contracts")
    public Contract createContract(@Valid @RequestBody Contract contract) {
        return contractRepo.save(contract);
    }

    @PutMapping("/contracts/{id}")
    public ResponseEntity<Contract> updateContract(@PathVariable(value = "id") Integer contractId,
                                               @Valid @RequestBody Contract contractDetails) throws ResourceNotFoundException {
        Contract contract= contractRepo.findById(contractId)
                .orElseThrow(() -> new ResourceNotFoundException("Contract not found"));
        contract.setContractType(contractDetails.getContractType());
        contract.setId(contractDetails.getId());
        contract.setPrice(contractDetails.getPrice());

        final Contract updatedContract = contractRepo.save(contract);
        return ResponseEntity.ok(updatedContract);
    }

    @DeleteMapping("/contracts/{id}")
    public Map<String, Boolean> deleteContract(@PathVariable(value = "id") Integer contractId)
            throws ResourceNotFoundException {
        Contract contract = contractRepo.findById(contractId)
                .orElseThrow(() -> new ResourceNotFoundException("Contract not found"));
        contractRepo.delete(contract);
        Map<String, Boolean> response = new HashMap<>();
        response.put("deleted", Boolean.TRUE);
        return response;
    }
}
